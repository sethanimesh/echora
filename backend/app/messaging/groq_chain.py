"""Post-ASR message assistance: one grounded LLM call over the beam alignment.

The recognizer returns several competing transcriptions of one utterance. Where
they disagree, that is almost always one sound heard several ways rather than
different words the speaker said. This module aligns the beams, asks a model to
resolve each position and write the sentence, and enforces exactly one property
in code: no content word the recognizer never produced.

Wording is left to the model. Earlier versions policed the English against a
word allowlist, which rejected "My leg hurts." and substituted a blunter
sentence; that lost the word that mattered while reporting success.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from groq import Groq

from ..config import Settings
from ..schemas import CommunicationContext, Hypothesis, MessageCandidate, RankerDecision


EMPTY_SLOT = "—"
MAX_OPTIONS = 3

CONTEXT_GUIDANCE: dict[CommunicationContext, str] = {
    "general": "No specific setting is known.",
    "home": "The speaker is at home.",
    "care": "The speaker is in a hospital or care setting.",
    "outdoors": "The speaker is outdoors.",
}

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

SYSTEM_PROMPT = """You help a stroke survivor be understood. They spoke ONE sentence. A speech recognizer that is often wrong on this voice returned several competing transcriptions of that same audio. You cannot hear the audio.

The alignment shows the utterance position by position. [a|b|c] means the recognizer heard one sound several ways. Pick the variant that forms ordinary English with its neighbours in this setting. Beam order is weak evidence: a contested sound splits one word across spellings, so the right word often looks like a minority.

Decide, for each contested position, which case you are in.

MISHEARING -- only one variant is a real word that fits the frame; the others are noise. Resolve it silently and return one message. In a care setting "[leg|link|league|lleg|ling] pain" is leg pain: only "leg" is a body part. Do not offer the others.

GENUINE CHOICE -- two or more variants are real words that fit, and a listener would act on them differently. Return one option for each, up to three. Examples that are a genuine choice: leg vs arm (different body part), tea vs toast (different thing to fetch), hot vs not (opposite meanings), window vs door (different object), head vs leg. When in doubt between MISHEARING and GENUINE CHOICE, it is a genuine choice -- the speaker confirms every message, and a wrong commit is harder to notice than an extra option.

NO PLAUSIBLE READING -- no combination of the recognizer's own words forms something a person would actually say. Set unclear=true and return the transcriptions as they are. Do NOT assemble a fluent sentence out of noise: inventing "I have a thin dog" from "fen tog / fend og / thin tog" is far worse than admitting the audio was not understood.

For each option return the reading (the recognizer's own words, one chosen per position, never a word absent from that position's options) and the message -- the sentence the speaker would have said if their speech were clear. First person, complete, natural, speakable, adult to adult. Rephrase freely for natural English: "leg pain" may become "My leg hurts."

Never make a message vaguer to be safe. If you resolve a position, keep that word: "I have pain" when they said "leg pain" drops the word that matters. Offering a real alternative is not vagueness -- dropping a word is.

The setting tells you which of the words the recognizer produced is most likely. It never lets you introduce a word no beam contains."""


@dataclass
class MessageChainResult:
    ranker: RankerDecision
    messages: list[MessageCandidate]
    ranking_seconds: float = 0.0
    grammar_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _tokens(text: str) -> list[str]:
    normalized = text.lower().replace("won't", "not").replace("can't", "cannot")
    return re.findall(r"[a-z0-9]+", normalized)


def _slot_alignment(hypotheses: list[Hypothesis]) -> list[dict[str, object]]:
    """Line the beams up against the leading beam, position by position."""
    reference = _tokens(hypotheses[0].literal_text)
    if not reference:
        return []
    total_weight = sum(item.search_weight for item in hypotheses) or 1.0
    columns: list[dict[str, float]] = [{} for _ in reference]
    for item in hypotheses:
        tokens = _tokens(item.literal_text)
        aligned: list[str] = [""] * len(reference)
        for tag, i1, i2, j1, j2 in SequenceMatcher(a=reference, b=tokens, autojunk=False).get_opcodes():
            if tag == "equal":
                for offset in range(i2 - i1):
                    aligned[i1 + offset] = tokens[j1 + offset]
            elif tag == "replace":
                span = tokens[j1:j2]
                for position in range(i1, i2):
                    offset = position - i1
                    aligned[position] = span[offset] if offset < len(span) else ""
        for position, token in enumerate(aligned):
            word = token or EMPTY_SLOT
            columns[position][word] = columns[position].get(word, 0.0) + item.search_weight / total_weight
    return [
        {
            "position": position + 1,
            "stable": len(column) == 1,
            "options": [
                {"word": word, "share": round(share, 3)}
                for word, share in sorted(column.items(), key=lambda pair: -pair[1])
            ],
        }
        for position, column in enumerate(columns)
    ]


def _alignment_template(slots: list[dict[str, object]]) -> str:
    parts = []
    for slot in slots:
        options = slot["options"]
        if slot["stable"]:
            parts.append(str(options[0]["word"]))
        else:
            parts.append("[" + "|".join(str(option["word"]) for option in options) + "]")
    return " ".join(parts)


def _slot_options(slots: list[dict[str, object]]) -> list[list[str]]:
    return [
        [str(option["word"]) for option in slot["options"] if option["word"] != EMPTY_SLOT]
        for slot in slots
    ]


def _word_alternatives(slots: list[dict[str, object]], reading: list[str]) -> dict[str, list[str]]:
    """Other words heard at each position the chosen reading used."""
    alternatives: dict[str, list[str]] = {}
    for slot, chosen in zip(slots, reading, strict=False):
        others = [
            str(option["word"])
            for option in slot["options"]
            if option["word"] not in (chosen, EMPTY_SLOT)
        ]
        if others:
            alternatives[chosen] = others
    return alternatives


def _grounded_reading(reading: str, slots: list[dict[str, object]]) -> list[str] | None:
    """The one property enforced in code: every word was actually heard.

    Checked positionally when the lengths agree, otherwise against the union of
    slot options. Returns None when nothing survives.
    """
    words = _tokens(reading)
    options = _slot_options(slots)
    if not words or not options:
        return None
    if len(words) == len(options):
        if all(word in allowed for word, allowed in zip(words, options, strict=True)):
            return words
        return None
    vocabulary = {word for allowed in options for word in allowed}
    kept = [word for word in words if word in vocabulary]
    return kept or None


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
    ) -> MessageChainResult:
        if not self.client:
            return _unavailable(hypotheses, "Groq is not configured; showing raw ASR candidates")
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(self._compose, hypotheses, context)
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
    ) -> MessageChainResult:
        slots = _slot_alignment(hypotheses)
        payload = {
            "setting": context,
            "setting_guidance": CONTEXT_GUIDANCE[context],
            "alignment": _alignment_template(slots),
            "slot_options": _slot_options(slots),
            "transcriptions": [item.literal_text for item in hypotheses],
        }
        model, body = self._complete(payload)
        if body.get("unclear"):
            return _unavailable(
                hypotheses,
                "The audio was not understood well enough to suggest a message; "
                "these are the raw transcriptions",
            )
        messages = self._build(body, hypotheses, slots)
        if not messages:
            return _unavailable(hypotheses, "No grounded message could be formed from the audio")
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
            ),
            messages=messages,
        )

    def _build(
        self,
        body: dict,
        hypotheses: list[Hypothesis],
        slots: list[dict[str, object]],
    ) -> list[MessageCandidate]:
        by_reading: dict[str, tuple[list[str], str]] = {}
        for option in body["options"]:
            message = option["message"].strip()
            reading = _grounded_reading(option["reading"], slots)
            if not message or not reading:
                continue
            by_reading.setdefault(" ".join(reading), (reading, message))

        # A reading contained by another is not a real alternative: offering both
        # "pain" and "leg pain" asks the speaker to choose between a message and a
        # worse version of the same message.
        kept = [
            (reading, message)
            for reading, message in by_reading.values()
            if not any(
                set(reading) < set(other)
                for other, _ in by_reading.values()
            )
        ]
        messages: list[MessageCandidate] = []
        for index, (reading, message) in enumerate(kept[:MAX_OPTIONS], 1):
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
                    repair_note=f"Heard as “{' '.join(reading)}”.",
                    word_alternatives=_word_alternatives(slots, reading),
                )
            )
        return messages

    def _call(self, model: str, payload: dict) -> dict:
        # `reasoning_effort: low` is a gpt-oss parameter; Qwen rejects it.
        extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_schema", "json_schema": MESSAGE_SCHEMA},
            **extra,
        )
        return json.loads(response.choices[0].message.content)

    def _complete(self, payload: dict) -> tuple[str, dict]:
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
                    return model, self._call(model, payload)
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
