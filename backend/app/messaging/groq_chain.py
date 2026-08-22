"""Post-ASR message assistance: one grounded LLM call over the beam alignment.

The recognizer returns several competing transcriptions of one utterance. Where
they disagree, that is almost always one sound heard several ways rather than
different words the speaker said. This module aligns the beams, asks a model to
resolve each position and write the sentence, and enforces exactly one property
in code: no content word the recognizer never produced.

Wording is left to the model, steered by who is speaking. Earlier versions
policed the English against a word allowlist, which rejected "My leg hurts."
and substituted a blunter sentence; that lost the word that mattered while
reporting success. But asking only for "first person" was just as wrong in the
other direction: it turned the instruction "clean the room" into "I need to
clean the room", fluent and grammatical and the opposite of what was meant.
This speaker is asking for help, reporting their body, or telling the person
with them what to do, so the prompt names those acts instead.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field

from groq import Groq

from ..config import Settings
from ..schemas import (
    CommunicationContext,
    Hypothesis,
    Listener,
    MessageCandidate,
    PersonalBrief,
    PersonalizationTrace,
    RankerDecision,
    Specialization,
)
from .alignment import (
    EMPTY_SLOT,
    _alignment_template,
    _grounded_reading,
    _slot_alignment,
    _slot_options,
    _tokens,
    settled_words,
    _word_alternatives,
)


MAX_OPTIONS = 3

# The setting says where the speaker is. On its own that only ever told the model
# which of the recognizer's words was likeliest -- it never changed what the
# message does. But "washroom" at home is a need stated to someone who can act on
# it, and "washroom" on a platform is a question put to someone who can only
# answer. The act is what changes, and the variable that decides it is who is
# listening, not where they are: two people can both be outdoors, one beside
# their daughter and one alone among strangers.
#
# So guidance is keyed by the pair. The four familiar stances say what they always
# said, with the act now stated rather than assumed. Only the unfamiliar one is
# new behaviour.
STANCE_GUIDANCE: dict[tuple[CommunicationContext, Listener], str] = {
    ("general", "familiar"): (
        "No specific setting is known; assume someone is there to help. "
        "Say what the speaker needs, as a statement to that person."
    ),
    ("home", "familiar"): (
        "The speaker is at home, talking to family or a carer in the house. "
        "This listener knows them and can fetch and do things for them, so a need "
        "is stated to them rather than asked of them."
    ),
    ("care", "familiar"): (
        "The speaker is in a hospital or care setting, talking to a nurse or "
        "carer. This listener is on duty and can act, so a need is stated to them "
        "rather than asked of them."
    ),
    ("outdoors", "familiar"): (
        "The speaker is out, with someone who knows them -- family, a friend, a "
        "carer who came along. That person can fetch and do things for them, so a "
        "need is stated to them exactly as it would be at home."
    ),
    ("outdoors", "unfamiliar"): (
        "The speaker is out among people who do not know them. This listener "
        "cannot fetch anything from their home and does not know their routine; "
        "they can answer, point, or serve. So a need is put to them as a question "
        "or a short request, not as a statement of what the speaker wants done: a "
        'place is asked for ("washroom" -> "Where is the washroom?"), a thing is '
        'asked for over a counter ("water" -> "Some water, please."), help is '
        'asked for directly ("help" -> "Could you help me?"). The question word is '
        "the form this act takes here, not new content the speaker did not supply. "
        "Keep it SHORTER than you would at home, never longer: a stranger is "
        "waiting, so the message carries the thing being asked for and nothing "
        "else. No softening, no explaining, no apologising for needing it."
    ),
    # A stranger at home or on the ward is a visitor, a delivery, a nurse the
    # speaker has not met. The setting still tells you what is around them; the
    # listener still cannot be sent to their kitchen.
    ("general", "unfamiliar"): (
        "No specific setting is known, and the person listening does not know the "
        "speaker. They can answer or pass something over, so put a need to them as "
        "a question or a short request, and keep it brief."
    ),
    ("home", "unfamiliar"): (
        "The speaker is at home but talking to someone who does not know them -- a "
        "visitor, a delivery, a carer on their first day. They do not know where "
        "anything is kept, so a need is put to them as a question or a short "
        "request, and kept brief."
    ),
    ("care", "unfamiliar"): (
        "The speaker is in a hospital or care setting, talking to someone who does "
        "not know them -- a nurse on a new rota, someone passing. Put a need to "
        "them as a question or a short request, and keep it brief."
    ),
}


def stance_guidance(context: CommunicationContext, listener: Listener) -> str:
    """The guidance for one stance. Falls back to the familiar reading of the setting."""
    return STANCE_GUIDANCE.get(
        (context, listener),
        STANCE_GUIDANCE.get((context, "familiar"), STANCE_GUIDANCE[("general", "familiar")]),
    )

MESSAGE_SCHEMA = {
    "name": "echora_message",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "options": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_OPTIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "reading": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["reading", "message"],
                    "additionalProperties": False,
                },
            },
            "unclear": {"type": "boolean"},
            "note": {"type": "string"},
        },
        "required": ["options", "unclear", "note"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You help a stroke survivor with dysarthria be understood by the person who is with them. They spoke ONE short utterance: something they need, something they feel, something they want that person to do, or an answer to what they were just asked. They are not making conversation and not narrating their day, and they cannot carry out physical tasks themselves. A speech recognizer that is often wrong on this voice returned several competing transcriptions of that same audio. You cannot hear the audio.

The alignment shows the utterance position by position. [a|b|c] means the recognizer heard one sound several ways. Pick the variant that forms ordinary English with its neighbours in this setting. Beam order is weak evidence: a contested sound splits one word across spellings, so the right word often looks like a minority.

Decide, for each contested position, which case you are in.

MISHEARING -- only one variant is a real word that fits the frame and the way an adult asks for help; the others are noise. Resolve it silently and return one message. In a care setting "[leg|link|league|lleg|ling] pain" is leg pain: only "leg" is a body part. "[blanket|blankie]" is blanket: a grown adult asking a nurse for bedding does not say blankie. Do not offer the others.

GENUINE CHOICE -- two or more variants are real words that fit, and a listener would act on them differently. Return one option for each, up to three. Examples that are a genuine choice: leg vs arm (different body part), tea vs toast (different thing to fetch), hot vs not (opposite meanings), window vs door (different object), head vs leg. When in doubt between MISHEARING and GENUINE CHOICE, it is a genuine choice -- a single option is spoken aloud the moment it arrives, with nothing to confirm, so a wrong commit is already said while an extra option only costs the speaker one tap.

NO PLAUSIBLE READING -- no combination of the recognizer's own words forms something this speaker would say to the person with them. Set unclear=true and return the transcriptions as they are. Grammatical English is not the test, and neither is a topic you can imagine someone discussing: "help wash [your|two|stool|tooot] room" has no variant that makes a need, a symptom, an instruction or an answer this speaker would send -- they are not asking a nurse to go and wash the nurse's own room -- so it stays unclear instead of becoming a tidy sentence about cleaning a room. A reading that comes out as the listener's own business is a sign that words were misheard, not a message. Do NOT assemble a fluent sentence out of noise: inventing "I have a thin dog" from "fen tog / fend og / thin tog" is far worse than admitting the audio was not understood.

For each option return the reading (the recognizer's own words, one chosen per position -- or none, where every variant of a position is noise -- never a word absent from that position's options) and the message -- what the speaker would have said to the person they are speaking to, if their speech were clear. Short, natural, speakable, adult to adult; someone in pain or waiting for help says one clause, not a paragraph. Rephrase freely for natural English, but keep the act the words carry. setting_guidance tells you who is listening and what form a message takes for them; where it disagrees with the three rules below, setting_guidance wins, because those rules describe someone who already knows this speaker and can act for them:
- a body part or a sensation is a report on themselves: "leg pain" -> "My leg hurts." "cold" -> "I am cold."
- a thing is a request for that thing: "blanket" -> "I need a blanket."
- an action is asked of the listener, never announced as the speaker's own plan: "clean the room" -> "Please clean the room." "call the nurse" -> "Please call the nurse." Not "I need to clean the room", not "I am going to call the nurse."
Supply nothing the speaker did not say: no reason, no apology, no explanation, no pleasantry. "Please" is the only word of politeness you may add. Putting a need to someone who does not know the speaker as a question is a change of form, not added content, so "where", "could" and "is there" are available for exactly that when setting_guidance calls for it -- and for nothing else.

Never make a message vaguer to be safe, and never pad it to sound complete. If you resolve a position, keep that word: "I have pain" when they said "leg pain" drops the word that matters. Offering a real alternative is not vagueness -- dropping a word is, and adding one the speaker never asked for puts words in their mouth.

The setting tells you which of the words the recognizer produced is most likely, and what form the message takes for the person listening. It never lets you introduce a word no beam contains."""


PERSONAL_MESSAGE_SCHEMA = {
    "name": "echora_personal_message",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "options": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_OPTIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "reading": {"type": "string"},
                        "message": {"type": "string"},
                        "specializations": {
                            "type": "array",
                            "maxItems": MAX_OPTIONS,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "anchor": {"type": "string"},
                                    "surface": {"type": "string"},
                                    "source": {"type": "string"},
                                },
                                "required": ["anchor", "surface", "source"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["reading", "message", "specializations"],
                    "additionalProperties": False,
                },
            },
            "unclear": {"type": "boolean"},
            "note": {"type": "string"},
        },
        "required": ["options", "unclear", "note"],
        "additionalProperties": False,
    },
}

PERSONAL_GUIDANCE = """

You also know some things about this particular speaker. They never change what counts as evidence.

known_words are people, places and things in this speaker's life. When a contested position offers one of them, that is strong evidence it is the right variant of that sound: "[marge|march|large] tea" from someone whose carer is Marge is Marge. This is still only a choice among the words the recognizer produced. It never lets you write a word no beam contains.

Several spellings in one position often name the same thing, and known_words tells you when: entries sharing a "means" are one option, not two. "[donna|dawn|dorn|danna]", where dorn means Dawn and danna means Donna, is a choice between two people, so it is two options and not four. Where every spelling in a position shares one "means" it is not a choice at all -- resolve it silently.

known_details are the speaker's own version of an ordinary thing. If your reading contains the anchor word, you may write say_instead in place of it -- copied exactly, not reworded -- and you must then list it under specializations with that anchor and its source. Never attach a detail to a word that is not in your reading, never invent a detail that is not listed, and never alter the wording you were given. An undeclared or altered detail is discarded and the plain wording is used instead.

A detail is not padding, so the instruction to keep a message short never trims one. It is *what* is being asked for: over a counter "a short black coffee, no sugar" is the order and "Some coffee, please." is the wrong drink, and a stranger who can hand over the right thing first time is exactly who this matters to. Shorten everything around a detail and leave the detail whole. If you decide against using one, do not list it under specializations either -- declaring a detail you did not write gets the message stripped back to the plain wording.

past_accepted_messages are messages this speaker settled on before, retrieved because they resemble this audio. They show you how this person phrases things and what they usually need. They are NOT evidence about what was said just now. Never take a word from them into your reading, and never answer with one of them because it looks close. Each one carries the setting and listener it was accepted under. Where those differ from this utterance's, the example still shows you the words this speaker uses -- it does not show you the form to use here. Follow setting_guidance for the form."""


@dataclass
class MessageChainResult:
    ranker: RankerDecision
    messages: list[MessageCandidate]
    ranking_seconds: float = 0.0
    grammar_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


# Articles and possessives carried by a display form like "the veranda" say
# nothing about whose word it is, and flagging them would reject ordinary English.
_AUDIT_IGNORED = frozenset(
    {"a", "an", "the", "my", "your", "his", "her", "their", "our", "some", "of", "to", "for"}
)


def _phrase_pattern(phrase: str) -> re.Pattern[str] | None:
    """Match a multi-word phrase on word boundaries, tolerating spacing and case."""
    words = _tokens(phrase)
    if not words:
        return None
    return re.compile(r"\b" + r"\W+".join(re.escape(word) for word in words) + r"\b", re.IGNORECASE)


def _replace_phrase(text: str, phrase: str, replacement: str) -> tuple[str, bool]:
    pattern = _phrase_pattern(phrase)
    if pattern is None or not pattern.search(text):
        return text, False
    return pattern.sub(replacement, text, count=1), True


def _apply_specializations(
    declared: list[dict],
    reading: list[str],
    message: str,
    slots: list[dict[str, object]],
    brief: PersonalBrief | None,
    anchor_share: float,
) -> tuple[list[Specialization], str, str | None, int]:
    """Keep only details the profile declared, riding on words that were heard.

    Returns the applied details, the message to show, the same message with every
    applied detail reverted, and how many declarations were refused. A refused
    declaration does not kill the message: the span the model marked as a detail
    is stripped back to the plain wording, which lands exactly where an
    un-personalized request would have.
    """
    applied: list[Specialization] = []
    refused = 0
    corrected = message
    settled = settled_words(slots, anchor_share)
    for record in declared:
        anchor_tokens = _tokens(str(record.get("anchor", "")))
        surface = str(record.get("surface", "")).strip()
        source = str(record.get("source", "")).strip()
        offer = brief.offer(anchor_tokens[0]) if (brief and len(anchor_tokens) == 1) else None
        legal = (
            offer is not None
            and len(anchor_tokens) == 1
            and source == offer.source
            and _tokens(surface) == _tokens(offer.surface)
            and anchor_tokens[0] in reading
            and anchor_tokens[0] in settled
            and _phrase_pattern(offer.surface) is not None
            and bool(_phrase_pattern(offer.surface).search(corrected))
        )
        if legal:
            applied.append(
                Specialization(
                    anchor=anchor_tokens[0],
                    plain=offer.plain,
                    surface=offer.surface,
                    source=offer.source,
                    kind=offer.kind,
                    profile_id=brief.profile_id,
                )
            )
            continue
        refused += 1
        # The model marked this span as a detail and we will not vouch for it, so
        # take it back out. Falling back to the profile's plain wording when we
        # have one, and to the bare heard word otherwise, keeps the sentence
        # readable instead of leaving invented decoration in the speaker's mouth.
        if surface:
            plain = offer.plain if offer else (anchor_tokens[0] if anchor_tokens else "")
            if plain:
                corrected, _ = _replace_phrase(corrected, surface, plain)
    plain_text: str | None = None
    if applied:
        reverted = corrected
        for detail in sorted(applied, key=lambda item: -len(item.surface)):
            reverted, _ = _replace_phrase(reverted, detail.surface, detail.plain)
        plain_text = reverted if reverted != corrected else None
    return applied, corrected, plain_text, refused


def _unlicensed_profile_words(
    message: str,
    slots: list[dict[str, object]],
    brief: PersonalBrief | None,
    applied: list[Specialization],
) -> list[str]:
    """Profile words in the message that neither a beam nor a licensed detail explains.

    This closes a gap the reading check cannot see. With beams `[march|large]` a
    reading of `march` and a message of "Ask Marge." passes grounding today,
    because the message is free prose. `marge` is a word only the profile
    supplied, no beam produced it, and no detail licensed it -- so it is caught
    here instead of being spoken as though the recognizer had heard it.
    """
    if brief is None or not brief.audit_vocabulary:
        return []
    profile_only = {word for word in brief.audit_vocabulary if word not in _AUDIT_IGNORED}
    if not profile_only:
        return []
    heard = {word for options in _slot_options(slots) for word in options}
    licensed: set[str] = set()
    for detail in applied:
        licensed |= set(_tokens(detail.surface))
    return [word for word in _tokens(message) if word in profile_only - heard - licensed]


def _is_rate_limited(error: Exception) -> bool:
    return getattr(getattr(error, "response", None), "status_code", None) == 429


def _quota_exhausted(error: Exception) -> bool:
    """True for a daily cap, false for a per-minute throttle.

    Groq enforces tokens-per-minute and tokens-per-day separately and reports
    both as 429. The per-minute bucket refills in about a second and must be
    waited out; the daily cap will not clear inside a retry loop and each attempt
    spends tokens the speaker may need for their next utterance.
    """
    if not _is_rate_limited(error):
        return False
    body = (getattr(getattr(error, "response", None), "text", "") or str(error)).lower()
    return "per day" in body or "tpd" in body or "rpd" in body


def _retry_after_seconds(error: Exception, default: float) -> float:
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    raw = headers.get("retry-after") or headers.get("x-ratelimit-reset-tokens") or ""
    match = re.match(r"^(?:(\d+)m)?([\d.]+)s?$", str(raw).strip())
    if match:
        minutes = float(match.group(1) or 0)
        return min(minutes * 60 + float(match.group(2)), 30.0)
    return default


def _recover_tool_call(error: Exception) -> dict | None:
    """Pull the answer out of a `tool_use_failed` 400, which quotes the generation."""
    body = getattr(getattr(error, "response", None), "text", "") or str(error)
    if "tool_use_failed" not in body:
        return None
    match = re.search(r'"failed_generation":\s*"((?:[^"\\]|\\.)*)"', body)
    if not match:
        return None
    try:
        parsed = json.loads(json.loads(f'"{match.group(1)}"'))
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("arguments"), dict):
        return parsed["arguments"]
    return parsed if isinstance(parsed, dict) else None


def _unavailable(hypotheses: list[Hypothesis], reason: str) -> MessageChainResult:
    """Show the raw beams, labelled as raw. Honest beats fluent when we cannot help."""
    messages = [
        MessageCandidate(
            message_id=f"m{index}",
            hypothesis_id=item.id,
            source_hypothesis_ids=[item.id],
            source_literals=[item.literal_text],
            literal_text=item.literal_text,
            interpreted_intent=item.literal_text,
            corrected_text=item.literal_text,
            repair_status="unavailable",
            repair_note="Message assistance is unavailable; this is raw ASR output.",
        )
        for index, item in enumerate(hypotheses[:MAX_OPTIONS], 1)
    ]
    return MessageChainResult(
        ranker=RankerDecision(
            decision="ambiguous",
            display_hypothesis_ids=[item.id for item in hypotheses],
            display_message_ids=[item.message_id for item in messages],
            reason=reason,
            source="unavailable",
        ),
        messages=messages,
        warnings=[reason],
    )


class GroqMessageChain:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None

    async def run(
        self,
        hypotheses: list[Hypothesis],
        context: CommunicationContext = "general",
        brief: PersonalBrief | None = None,
        listener: Listener = "familiar",
    ) -> MessageChainResult:
        if not self.client:
            return _unavailable(hypotheses, "Groq is not configured; showing raw ASR candidates")
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(self._compose, hypotheses, context, brief, listener)
        except Exception as error:
            result = _unavailable(
                hypotheses,
                "The message assistant is over its daily usage limit; showing raw ASR candidates"
                if _quota_exhausted(error)
                else f"Message assistance failed; showing raw ASR candidates ({type(error).__name__})",
            )
        result.ranking_seconds = round(time.perf_counter() - started, 3)
        return result

    def _compose(
        self,
        hypotheses: list[Hypothesis],
        context: CommunicationContext,
        brief: PersonalBrief | None = None,
        listener: Listener = "familiar",
    ) -> MessageChainResult:
        slots = _slot_alignment(hypotheses)
        # The evidence keys stay first and unchanged. Everything personal is
        # appended after them, so the model reads what was heard before it reads
        # anything about who was speaking, and an un-personalized request sends
        # byte-for-byte what it sent before this layer existed.
        payload = {
            "setting": context,
            "setting_guidance": stance_guidance(context, listener),
            "alignment": _alignment_template(slots),
            "slot_options": _slot_options(slots),
            "transcriptions": [item.literal_text for item in hypotheses],
        }
        personalized = brief is not None and not brief.is_empty()
        if personalized and brief is not None:
            if brief.speaker_note:
                payload["speaker"] = brief.speaker_note
            if brief.lexicon:
                payload["known_words"] = [
                    {"heard": hint.word, "means": hint.display, "kind": hint.kind, "note": hint.note}
                    for hint in brief.lexicon
                ]
            if brief.specializations:
                payload["known_details"] = [
                    {"anchor": offer.anchor, "say_instead": offer.surface, "source": offer.source}
                    for offer in brief.specializations
                ]
            if brief.examples:
                payload["past_accepted_messages"] = [
                    {
                        "heard": example.heard,
                        "message": example.message,
                        "setting": example.context,
                        "listener": example.listener,
                    }
                    for example in brief.examples
                ]
        model, body = self._complete(payload, personalized)
        if body.get("unclear"):
            return _unavailable(
                hypotheses,
                "The audio was not understood well enough to suggest a message; "
                "these are the raw transcriptions",
            )
        messages, refused = self._build(body, hypotheses, slots, brief if personalized else None)
        if not messages:
            return _unavailable(hypotheses, "No grounded message could be formed from the audio")
        trace: PersonalizationTrace | None = None
        if personalized and brief is not None:
            trace = PersonalizationTrace(
                profile_id=brief.profile_id,
                profile_label=brief.profile_label,
                lexicon_hints=[hint.word for hint in brief.lexicon],
                examples_used=len(brief.examples),
                specializations_offered=len(brief.specializations),
                specializations_applied=sum(len(item.specializations) for item in messages),
                specializations_refused=refused,
                retrieval_seconds=brief.retrieval_seconds,
            )
        return MessageChainResult(
            ranker=RankerDecision(
                decision="selected" if len(messages) == 1 else "ambiguous",
                selected_message_id=messages[0].message_id if len(messages) == 1 else None,
                display_hypothesis_ids=[item.id for item in hypotheses],
                display_message_ids=[item.message_id for item in messages],
                reason=(
                    body["note"].strip()
                    if len(messages) == 1
                    else "These could be different messages; please choose."
                ),
                source="groq",
                assistant_model=model,
                personalization=trace,
            ),
            messages=messages,
        )

    def _build(
        self,
        body: dict,
        hypotheses: list[Hypothesis],
        slots: list[dict[str, object]],
        brief: PersonalBrief | None = None,
    ) -> tuple[list[MessageCandidate], int]:
        allow_details = brief is not None and self.settings.personal_specializations
        by_reading: dict[str, tuple[list[str], str, list[Specialization], str | None]] = {}
        refused = 0
        for option in body["options"]:
            message = option["message"].strip()
            reading = _grounded_reading(option["reading"], slots)
            if not message or not reading:
                continue
            applied: list[Specialization] = []
            plain: str | None = None
            declared = option.get("specializations") or []
            if declared and allow_details:
                applied, message, plain, rejected = _apply_specializations(
                    declared, reading, message, slots, brief, self.settings.personal_anchor_share
                )
                refused += rejected
            elif declared:
                # The detail layer is switched off, so nothing may claim profile
                # provenance; the wording still gets stripped back to plain.
                _, message, _, rejected = _apply_specializations(
                    declared, reading, message, slots, None, self.settings.personal_anchor_share
                )
                refused += rejected
            # A profile word that no beam produced and no licensed detail explains
            # is treated exactly like an ungrounded reading: the option is dropped.
            if _unlicensed_profile_words(message, slots, brief, applied):
                continue
            by_reading.setdefault(" ".join(reading), (reading, message, applied, plain))

        # A reading contained by another is not a real alternative: offering both
        # "pain" and "leg pain" asks the speaker to choose between a message and a
        # worse version of the same message.
        kept = [
            entry
            for entry in by_reading.values()
            if not any(set(entry[0]) < set(other[0]) for other in by_reading.values())
        ]
        messages: list[MessageCandidate] = []
        for index, (reading, message, applied, plain) in enumerate(kept[:MAX_OPTIONS], 1):
            literals = [
                item.literal_text
                for item in hypotheses
                if set(reading) & set(_tokens(item.literal_text))
            ] or [hypotheses[0].literal_text]
            messages.append(
                MessageCandidate(
                    message_id=f"m{index}",
                    hypothesis_id=hypotheses[0].id,
                    source_hypothesis_ids=[item.id for item in hypotheses],
                    source_literals=literals,
                    literal_text=hypotheses[0].literal_text,
                    interpreted_intent=" ".join(reading),
                    corrected_text=message,
                    repair_status="corrected" if _tokens(message) != reading else "unchanged",
                    repair_note=f"Heard as \u201c{' '.join(reading)}\u201d.",
                    word_alternatives=_word_alternatives(slots, reading),
                    specializations=applied,
                    plain_text=plain,
                )
            )
        return messages, refused

    def _call(self, model: str, payload: dict, personalized: bool = False) -> dict:
        # `reasoning_effort: low` is a gpt-oss parameter; Qwen rejects it.
        extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
        prompt = SYSTEM_PROMPT + PERSONAL_GUIDANCE if personalized else SYSTEM_PROMPT
        schema = PERSONAL_MESSAGE_SCHEMA if personalized else MESSAGE_SCHEMA
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_schema", "json_schema": schema},
            **extra,
        )
        return json.loads(response.choices[0].message.content)

    def _complete(self, payload: dict, personalized: bool = False) -> tuple[str, dict]:
        """Ask the first model that answers, walking the configured chain.

        Groq meters each model separately, so a rate limit is a reason to switch
        models rather than to wait -- switching is faster than the per-minute
        bucket refilling, and the speaker is waiting on this call. gpt-oss also
        intermittently returns the answer as a rejected tool call, which is
        recovered from the error body or retried once on the same model.
        """
        last: Exception | None = None
        for model in self.settings.groq_models:
            for attempt in range(2):
                try:
                    return model, self._call(model, payload, personalized)
                except Exception as error:
                    recovered = _recover_tool_call(error)
                    if recovered is not None:
                        return model, recovered
                    last = error
                    if _is_rate_limited(error):
                        break  # this model's bucket is spent; try the next one
                    if attempt == 0:
                        time.sleep(0.4)
        raise last if last else RuntimeError("no Groq model configured")
