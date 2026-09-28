"""Lexical fast path: decide what can be decided without an LLM.

This stage exists to keep the LLM off the hot path. It resolves the large
majority of tokens from two evidence sources and, crucially, **refuses to guess**
when they disagree.

THE ARBITRATION RULE
--------------------
Two independent signals per token:

* *Hindi evidence* -- the token is a key in the Dakshina-derived Roman-Hindi
  lexicon.
* *English evidence* -- ``wordfreq`` Zipf frequency in English is at or above
  :data:`EN_ZIPF_STRONG`.

The mapping is deliberately asymmetric and refuses to break ties:

===================  ===================  ==================
Hindi evidence       English evidence     Result
===================  ===================  ==================
yes                  no                   ``HI``
no                   yes                  ``EN``
yes                  yes                  ``EN`` if the decisive-EN gate
                                          fires, else ``AMBIGUOUS`` -> LLM
no                   no                   ``AMBIGUOUS`` -> LLM
===================  ===================  ==================

The both-yes row is the entire reason this design needs an LLM at all. ``main``,
``is``, ``to``, ``me``, ``par``, ``or``, ``do``, ``bar``, ``so``, ``the``, ``hum``
are simultaneously frequent English words and common Roman-Hindi words; only
sentence context separates them, and no frequency threshold can. Letting the
higher score win would silently mistranslate one language into the other in
roughly half of all cases, which is why neither signal is permitted to
outrank the other.

The neither-yes row matters just as much: an unknown token is *unknown*, not
English. Defaulting OOV tokens to EN would leave every informal Hinglish
spelling the lexicon lacks ("bohot", "kyu", "nhi") unconverted.

LOANWORDS
---------
Dakshina contains English loanwords written in Devanagari -- ``office`` maps to
ऑफिस, ``meeting`` to मीटिंग. Without the both-yes rule these would transliterate,
turning "office" into "ऑफिस" in an otherwise English-preserving pipeline.

Sending all of them to the LLM instead is correct but far too expensive: they
are thousands of ordinary English words, and the model is *worse* at them than a
threshold is (qwen3:4b labels "attend" HI). The decisive-EN gate --
:data:`MIN_EN_LEN`, :data:`EN_ZIPF_DECISIVE`, :data:`HI_MARGIN_VETO` -- resolves
the safe interior of that pool without a call, and leaves the genuinely
either-way tokens ambiguous. Its measurements and its limits are documented with
the constants; read those before touching any threshold, because the gold set
cannot detect the failure mode the gate is guarding against.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from wordfreq import zipf_frequency

from communication.backend.hinglish_core.core.model import Label, Segment, Source, Utterance
from communication.backend.hinglish_core.detect.spans import acronym_hint

# The English-evidence bar, chosen from measurement rather than intuition.
#
# Dakshina contains thousands of English loanwords written in Devanagari
# (laptop -> लैपटॉप, password -> पासवर्ड, charger -> चार्जर). Any English word
# below this threshold that is *also* a lexicon key gets classified HI and
# silently transliterated -- a severe, silent corruption of English text.
#
# Measured on a 30-word Hindi set and a 25-word English/technical set:
#
#   threshold   Hindi words made ambiguous   English loanwords still corrupted
#   4.5                 0/30                         13   <- unacceptable
#   4.0                 0/30                          6
#   3.5                 1/30                          0   <- chosen
#   3.0                 4/30                          0
#
# The two distributions separate cleanly: Roman-Hindi tokens top out at 3.62
# ("tha") and English loanwords start at 3.64 ("charger"). 3.5 sits in that gap,
# eliminating the corruption for the cost of one extra ambiguous token in thirty.
# Raising this value trades correctness for a marginally cheaper LLM bill --
# do not do it without re-running the measurement.
EN_ZIPF_STRONG = 3.5

# Tokens shorter than this are too collision-prone to decide lexically at all.
MIN_DECIDABLE_LEN = 2

# Tokens that are common in BOTH languages but whose English frequency still
# falls below EN_ZIPF_STRONG, so the threshold alone would hand them to Hindi.
#
# This list is the residue left after the threshold was tuned: 3.5 handles the
# loanword problem in bulk, and these are the stragglers below it. "hum" (3.59)
# is the clearest case -- an ordinary English verb and also हम ("we"). Short
# three-letter tokens dominate because they collide most easily across scripts.
#
# Keep this list small and hand-checked. It is a targeted correction, not a
# substitute for the threshold; add to it when a false HI is observed in real
# use rather than pre-emptively.
COLLISION_TOKENS = frozenset({
    "par", "hum", "kar", "jab", "tab", "ban", "gale", "dal", "sab", "pal",
    "dam", "din", "log", "man", "chal", "bane", "mare", "sane", "tune",
    "sale", "bare", "kane", "mane", "pane", "bore", "core", "dare",
})

# --- The decisive-EN gate -------------------------------------------------
#
# The both-yes row above is correct in principle and far too broad in practice.
# 3,318 lexicon keys have English Zipf >= 4.0, and the overwhelming majority are
# ordinary English words that Dakshina happens to also carry as Devanagari
# loanwords (office -> ऑफिस, meeting -> मीटिंग, attend -> अटेंड). Sending them to
# the LLM costs latency and money, and measurably costs *accuracy*: qwen3:4b
# labels "attend" HI, which is simply wrong.
#
# This gate decides EN outright for the safe interior of that pool. It is a
# COLLISION-RISK gate, not a language discriminator -- it does not claim to know
# which language a token belongs to, only that the token is far enough from any
# plausible collision to stop paying for context. Three independent filters:
#
# 1. LENGTH. Collision rate falls off a cliff between 4 and 5 characters,
#    measured over the pool as "top Devanagari reading is more frequent in Hindi
#    than the Roman string is in English":
#
#      len 1  40.0% (25)    len 4   9.6% (627)    len 7  3.5% (453)
#      len 2  37.5% (104)   len 5   4.7% (620)    len 8  1.4% (284)
#      len 3  24.2% (269)   len 6   4.9% (569)    len 9+ 1.1% (367)
#
#    Every genuine short collision (main, par, chal, sale, din, log) is len <= 4
#    or already in COLLISION_TOKENS. Length is not a linguistic signal and is not
#    meant to be one; it is a direct proxy for collision risk, which is the thing
#    actually being gated. Do not "improve" this into a phonetic loanword
#    detector -- that was measured and it is refuted. Because the lexicon maps
#    roman -> Devanagari, every entry is by construction a romanisation of its
#    own Devanagari, so round-tripping measures how regularly the *English* word
#    is spelled, not which language it belongs to. meeting -> मीटिंग -> "miting"
#    scores identically to the true collision main -> मैं -> "main".
#
# 2. FREQUENCY. A higher bar than EN_ZIPF_STRONG. 3.5 answers "is there any
#    English evidence at all"; 4.0 answers "is there enough to decide without
#    context". The nearest genuine Roman-Hindi word below the line is "niche"
#    (नीचे) at 3.85, so headroom is 0.15 -- thin, which is why filter 3 exists.
#
# 3. HINDI-SIDE MARGIN. The one signal the length+frequency rule cannot see:
#    wordfreq also ships a Hindi list, so the Devanagari reading has a frequency
#    too. en_zipf - hi_zipf separates loanwords from native Hindi words cleanly
#    where length and English frequency alone do not:
#
#      karen  करें   en 4.09  hi 6.06  margin -1.97   native Hindi
#      maine  मैंने  en 4.16  hi 5.72  margin -1.56   native Hindi
#      thick  ठीक    en 4.46  hi 5.49  margin -1.03   native Hindi
#      banana बनाना  en 4.00  hi 5.03  margin -1.03   native Hindi
#      meeting मीटिंग en 5.15  hi 4.35  margin +0.80   loanword
#      attend अटेंड  en 4.55  hi 0.00  margin +4.55   loanword
#
#    Without this veto the gate forces maine, karen and banana to EN -- three of
#    the most frequent words in spoken Hinglish, and three the shipped model
#    currently gets RIGHT. The veto costs 37 of 2,293 pool tokens (1.6%) and zero
#    gold spans. It also independently protects "Delhi" (margin -1.59) and other
#    Indian proper nouns, which the mid-sentence capitalisation deferral would
#    otherwise be solely responsible for.
#
# Measured on the 90-sentence gold set (tests/eval/gold.jsonl), pool = lexicon
# keys the gate can newly decide, "wrong" = gold spans given the wrong label:
#
#   config                              pool  gold_amb  resolved  wrong  breaks
#   baseline (no gate)                     -        99         0      0  -
#   len>=5 zipf>=4.0                    2293        83        16      0  maine karen banana
#   len>=5 zipf>=4.5                    1113        85        14      0  chain cheese earth
#   len>=5 zipf>=4.0 margin>=-0.5       2275        83        16      0  chain cheese
#   len>=5 zipf>=4.0 margin>=-0.5 +exc  2256        83        16      0  none   <- chosen
#
# Note the gold set reports 0 broken labels at EVERY setting, including the
# unsafe zipf>=4.0-alone row, because it contains none of the counterexamples.
# A green benchmark is NOT evidence for this gate. The margin veto and the
# exception list are what make it safe; the benchmark cannot see either.
MIN_EN_LEN = 5
EN_ZIPF_DECISIVE = 4.0
HI_MARGIN_VETO = -0.5

# The mirror image of COLLISION_TOKENS, and deliberately a separate constant:
# COLLISION_TOKENS holds English words whose frequency falls BELOW the evidence
# bar, whereas these are Hindi words whose English twin sits ABOVE the decisive
# bar. Folding them together would make that list's docstring false.
#
# These are the residue the margin veto cannot catch, where Hindi and English
# frequency are too close for the margin to separate them but a Hinglish speaker
# would still be typing Hindi:
#
#   chain   चैन   en 4.65 hi 4.52  "chain se baith jao" (peace/rest)
#   cheese  चीज़   en 4.57 hi 4.72  "ye cheese wapas chahiye" (thing)
#   niche   नीचे  en 3.85 hi 5.45  below the zipf bar today, but the closest
#                                  genuine Hindi word to it -- listed on merit
#                                  so a corpus refresh cannot silently admit it
#
# The remaining four are already protected by the margin veto with >= 0.5 zipf
# of headroom. They are repeated here as belt-and-braces because they are the
# highest-frequency Hindi words in the affected pool and the thresholds above
# are derived from one wordfreq version:
#
#   maine मैंने, karen करें, banana बनाना, thick ठीक
#
# NOT listed, deliberately: Dakshina romanisation noise such as earth -> अर्थ,
# suffer -> सफर, stick -> सटीक, charm -> चरम, shock -> शोक, companion -> कंपनियों.
# The Devanagari is native Hindi but nobody types the Roman that way -- a user
# who writes "earth" means earth -- so forcing EN on these is correct, and
# exempting them would only send real English words back to the LLM.
DECISIVE_EN_EXCEPTIONS = frozenset({
    "chain", "cheese", "niche", "maine", "karen", "banana", "thick",
})


@lru_cache(maxsize=1)
def load_lexicon(path: str | None = None) -> dict[str, list[str]]:
    """Load the shipped Roman->Devanagari lexicon. Cached for process lifetime.

    ~13 ms and ~2 MB resident; loaded lazily so importing the package stays cheap
    for callers that only want segmentation.
    """
    if path is not None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        resource = files("communication.backend.hinglish_core.data").joinpath("hi_roman2dev.json")
        data = json.loads(resource.read_text(encoding="utf-8"))
    return data["lexicon"]


class LexicalClassifier:
    """Assigns HI / EN / AMBIGUOUS using lexicon + frequency evidence.

    ``protected_terms`` is the application-specific allowlist: names, brands and
    jargon that must never be transliterated regardless of what the lexicon
    says. It is checked first and wins outright, because it encodes deliberate
    operator knowledge that beats any statistical signal -- and because it is the
    one place a user of the device can correct a persistent misclassification
    without touching code.
    """

    def __init__(
        self,
        lexicon: dict[str, list[str]] | None = None,
        protected_terms: set[str] | None = None,
        en_zipf_strong: float = EN_ZIPF_STRONG,
    ) -> None:
        self._lexicon = lexicon if lexicon is not None else load_lexicon()
        self._protected = {t.lower() for t in (protected_terms or set())}
        self._en_zipf = en_zipf_strong

    def hindi_candidates(self, token: str) -> list[str]:
        """Devanagari readings for ``token``, best first. Empty if unknown."""
        return self._lexicon.get(token.lower(), [])

    def english_zipf(self, token: str) -> float:
        return zipf_frequency(token.lower(), "en")

    def hindi_zipf(self, candidates: tuple[str, ...] | list[str]) -> float:
        """Frequency of the most common Devanagari reading. 0.0 if unknown.

        Takes the max rather than the first candidate: the lexicon orders by
        romanisation confidence, not by Hindi frequency, so the first reading is
        not necessarily the one a speaker means.
        """
        return max((zipf_frequency(c, "hi") for c in candidates), default=0.0)

    def _decides_english(self, lower: str, candidates: list[str]) -> bool:
        """The decisive-EN gate. See the constants above for the measurements.

        Only ever consulted in the both-evidence branch, so it is structurally
        incapable of overturning an HI decision -- it can only convert AMBIGUOUS
        to EN, never the reverse.
        """
        if len(lower) < MIN_EN_LEN:
            return False
        # Redundant today (every COLLISION_TOKEN is <= 4 characters, so the
        # length gate already excludes them) but kept so that adding a longer
        # collision token later is honoured without editing this function.
        if lower in COLLISION_TOKENS or lower in DECISIVE_EN_EXCEPTIONS:
            return False
        en = self.english_zipf(lower)
        if en < EN_ZIPF_DECISIVE:
            return False
        return (en - self.hindi_zipf(candidates)) >= HI_MARGIN_VETO

    def classify_token(
        self, token: str, sentence_initial: bool = True
    ) -> tuple[Label, Source, tuple[str, ...]]:
        """Classify one token. Returns (label, source, candidates).

        ``sentence_initial`` suppresses the proper-noun signal, since the first
        word of a sentence is capitalised regardless of what it is. Defaults to
        True (the conservative reading) so a caller that lacks position
        information never gets a spurious NAME deferral.
        """
        lower = token.lower()

        if lower in self._protected:
            return Label.NAME, Source.ALLOWLIST, ()

        # Already Devanagari: nothing to do, and definitely not English.
        if any(0x0900 <= ord(c) <= 0x097F for c in token):
            return Label.HI, Source.LEXICON, ()

        if len(token) < MIN_DECIDABLE_LEN:
            return Label.AMBIGUOUS, Source.LEXICON, ()

        # A capitalised token in mid-sentence is a likely proper noun. Many
        # Indian names are also ordinary Hindi words with low English frequency
        # -- Aman (अमन "peace"), Ravi (रवि "sun"), Karan (करण) -- so the fast
        # path would otherwise transliterate them with total confidence and they
        # would never reach the LLM at all.
        #
        # This defers to context rather than deciding NAME: measured on the gold
        # set the signal is only ~71% precise for NAME (it also catches "Mast"
        # after an exclamation mark), which is far too weak to decide on, but
        # perfectly good as a reason to stop guessing. Cost is ~4% more LLM
        # traffic.
        #
        # LIMITATION: sentence-initial names carry no signal at all -- 8 of the
        # 13 names in the gold set are the first word and stay broken. The
        # remedy for those is the ``protected_terms`` allowlist, which is the
        # right mechanism anyway for a single-user device whose contacts, city
        # and regular brands are known and stable.
        if not sentence_initial and token[:1].isupper():
            return Label.AMBIGUOUS, Source.LEXICON, tuple(
                self.hindi_candidates(lower)
            )

        candidates = self.hindi_candidates(lower)
        has_hi = bool(candidates)
        # Three independent routes to "English evidence". Any one of them is
        # enough to block an unchecked HI decision:
        #   1. Ordinary frequency evidence.
        #   2. A curated collision token, whose English frequency sits below the
        #      threshold but which is still a real English word.
        #   3. An acronym shape. "API" is a lexicon key (एपीआई) with English Zipf
        #      3.68, so without this it would transliterate. Acronyms are not
        #      structurally decidable -- "KYA HUA BHAI" has the same shape -- so
        #      this is a hint that forces context, never a decision.
        has_en = (
            self.english_zipf(lower) >= self._en_zipf
            or lower in COLLISION_TOKENS
            or acronym_hint(token)
        )

        if has_hi and not has_en:
            return Label.HI, Source.LEXICON, tuple(candidates)
        if has_en and not has_hi:
            return Label.EN, Source.LEXICON, ()

        # Both: the collision row. Most of it is not actually colliding -- it is
        # ordinary English that Dakshina also carries as a Devanagari loanword.
        # The gate skims off the part that is far enough from any plausible
        # collision to decide without context.
        #
        # Placement is deliberate. This sits AFTER the capitalisation deferral so
        # a mid-sentence proper noun is still handed to the LLM rather than being
        # decided here. The margin veto happens to protect the names measured on
        # the gold set ("Delhi") on its own, but that is a second line of
        # defence, not a licence to hoist this check above the deferral.
        if has_hi and self._decides_english(lower, candidates):
            return Label.EN, Source.LEXICON, ()

        # Both (and still colliding) or neither: refuse to guess. Carry
        # candidates so that if the LLM later says HI, transliteration needs no
        # second lookup.
        return Label.AMBIGUOUS, Source.LEXICON, tuple(candidates)

    def run(self, utterance: Utterance) -> Utterance:
        """Label every word segment. Structural segments are left untouched."""
        out: list[Segment] = []
        first_word_start = next(
            (s.start for s in utterance.segments if s.is_word), None
        )
        for seg in utterance.segments:
            if not seg.is_word or seg.source is Source.STRUCTURAL:
                out.append(seg)
                continue
            label, source, candidates = self.classify_token(
                seg.text, sentence_initial=(seg.start == first_word_start)
            )
            out.append(seg.decided(label, source, candidates=candidates))
        return utterance.with_segments(out)


def ambiguous_spans(utterance: Utterance) -> list[Segment]:
    """The segments that still need contextual resolution."""
    return [s for s in utterance.segments if s.label is Label.AMBIGUOUS]
