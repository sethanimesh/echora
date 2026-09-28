"""The shared classification prompt and JSON schema.

Both the Ollama and Groq adapters use these verbatim. That is deliberate: if the
prompt differed per vendor, a fallback from local to cloud could silently change
the *meaning* of a label, and benchmark numbers from one vendor would not
transfer to the other.

EVERY BYTE HERE IS SENT TO THE MODEL ON EVERY CALL. Keep engineering notes in
Python comments like this one, never inside ``SYSTEM_PROMPT``. Appending a
twelve-line rationale into the prompt string cost 4.2 points of accuracy on the
gold set (90.5% -> 86.3%) before it was caught.

ON THE ``confidence`` FIELD (measured; do not "improve" it without re-running
the benchmark): a 4B model does not calibrate it. With the current wording it
answers "unsure" for essentially every span, including the ones it gets right.
Rewriting rule 4 to define "certain" and demand deliberate use flipped it to
"certain" for every span -- including "main" in "The main office is closed",
which is confidently *wrong* -- and cost 3.6 points (90.5% -> 86.9%). Both
wordings are equally uninformative and the current one is cheaper, so it stays.
Consequence: treat ``confidence`` as unusable on small local models, and do not
enable ``ECHORA_ESCALATE_ON_UNSURE`` against one -- it would escalate every
span, adding a second model call without a useful confidence threshold.
"""

from __future__ import annotations

import json

from communication.backend.hinglish_core.classify.port import LLM_LABELS, SpanQuery

SYSTEM_PROMPT = """\
You label words in code-mixed Hinglish (Hindi written in Roman script, mixed \
with English) for a text-to-speech system.

You will be given a sentence and a numbered list of spans taken from it. For \
each span, return exactly one label:

- HI: a Hindi word written in Roman script (it would be written in Devanagari \
by a Hindi speaker). Examples: main (मैं), kal, mein, karunga, nahi, kya.
- EN: an English word, including English loanwords used in Hinglish speech. \
Examples: office, meeting, important, laptop, deploy.
- NAME: a proper noun - a person, place, brand or product. Examples: Aman, \
Delhi, Zomato, Paytm.
- NUMBER: numeric content.
- CODE_URL: a URL, email address, file path, or code identifier.
- OTHER: punctuation, an interjection, or genuinely undecidable.

CRITICAL RULES
1. You are labelling ONLY. Never translate, rewrite, transliterate, correct \
spelling, or produce any Devanagari text. Return labels and nothing else.
2. Use the surrounding sentence to decide. Many tokens are BOTH common English \
words AND common Hindi words, and only context separates them:
   - "main": HI in "Main ghar ja raha hoon" (मैं = I); EN in "the main office".
   - "is": HI in "is kaam ko dekho" (इस = this); EN in "this is good".
   - "to": HI in "tum to jaante ho" (तो); EN in "I went to Delhi".
   - "me": HI in "ghar me hoon" (में = in); EN in "give me that".
   - "par": HI in "table par rakho" (पर = on); EN in "par for the course".
   - "or": HI in "tum or main" (और = and); EN in "tea or coffee".
   - "do": HI in "do minute" (दो = two); EN in "do it now".
   - "bar": HI in "ek bar" (बार = time); EN in "at the bar".
3. If a word is English by origin but is being used inside a Hindi clause as an \
ordinary borrowed word, label it EN. Echora preserves English words as-is, and \
that is the correct pronunciation for a loanword.
4. When genuinely uncertain, set confidence to "unsure" rather than guessing \
confidently. Do not force a token into HI or EN when OTHER is honest.

Return one entry per span, using the span ids exactly as given."""

SENSE_SELECTION_PROMPT = """\


Some spans below also have a numbered, closed list of Devanagari readings from \
the lexicon. When such a span is HI, use the sentence context to choose its \
intended reading and return that zero-based candidate_index. For every other \
span return -1. Never copy or emit a candidate's text; return only its index."""


def uses_candidate_selection(spans: list[SpanQuery]) -> bool:
    """Whether this request needs the conditional sense-selection contract."""
    return any(len(span.candidates) > 1 for span in spans)


def system_prompt(spans: list[SpanQuery]) -> str:
    """Keep the benchmarked label-only prompt unchanged unless sense is needed."""
    if uses_candidate_selection(spans):
        return SYSTEM_PROMPT + SENSE_SELECTION_PROMPT
    return SYSTEM_PROMPT


def build_user_message(sentence: str, spans: list[SpanQuery]) -> str:
    """Render the per-request half of the prompt.

    Kept short and placed *after* the system prompt so that prompt-caching, where
    a vendor supports it, can reuse the long stable prefix.
    """
    with_candidates = uses_candidate_selection(spans)
    lines: list[str] = []
    for span in spans:
        line = f"{span.span_id}. {span.text}"
        if with_candidates and span.candidates:
            choices = ", ".join(
                f"{index}={candidate}"
                for index, candidate in enumerate(span.candidates)
            )
            line += f" | candidates: {choices}"
        lines.append(line)
    listing = "\n".join(lines)
    return f"Sentence: {sentence}\n\nSpans to label:\n{listing}"


def response_schema(spans: list[SpanQuery]) -> dict:
    """JSON Schema constraining the reply to labels only.

    Every field is an enum or an integer; no field accepts free text. Combined
    with constrained decoding (Ollama ``format``, Groq strict ``json_schema``),
    a rewritten sentence is not merely discouraged -- it is unrepresentable.
    """
    item_properties = {
        "id": {"type": "integer"},
        "label": {"type": "string", "enum": list(LLM_LABELS)},
        "confidence": {
            "type": "string",
            "enum": ["certain", "unsure"],
        },
    }
    item_required = ["id", "label", "confidence"]
    if uses_candidate_selection(spans):
        max_candidates = max(len(span.candidates) for span in spans)
        item_properties["candidate_index"] = {
            "type": "integer",
            "enum": [-1, *range(max_candidates)],
        }
        item_required.append("candidate_index")

    return {
        "type": "object",
        "properties": {
            "spans": {
                "type": "array",
                "minItems": len(spans),
                "maxItems": len(spans),
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": item_required,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["spans"],
        "additionalProperties": False,
    }


def parse_response(
    payload: str | dict, spans: list[SpanQuery]
) -> dict[int, tuple[str, str, int | None]]:
    """Parse a reply into ``{span_id: (label, confidence, candidate_index)}``.

    Tolerates a model that returns extra, missing or unknown-id entries -- the
    caller supplies defaults for anything absent. Being lenient here is right:
    a malformed reply should degrade to the fallback policy, not crash the
    device mid-sentence.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    by_id = {span.span_id: span for span in spans}
    out: dict[int, tuple[str, str, int | None]] = {}
    for entry in data.get("spans", []):
        if not isinstance(entry, dict):
            continue
        span_id = entry.get("id")
        label = entry.get("label")
        if span_id in by_id and label in LLM_LABELS:
            confidence = entry.get("confidence")
            candidate_index = entry.get("candidate_index")
            candidates = by_id[span_id].candidates
            if (
                not isinstance(candidate_index, int)
                or isinstance(candidate_index, bool)
                or not 0 <= candidate_index < len(candidates)
            ):
                candidate_index = None
            out[span_id] = (
                label,
                confidence if confidence in ("certain", "unsure") else "certain",
                candidate_index,
            )
    return out
