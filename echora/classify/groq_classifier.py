"""Cloud span classifier backed by Groq. The configured fallback.

Reached when an earlier classifier is unavailable or, when explicitly enabled,
reports low confidence.
"""

from __future__ import annotations

import os

from echora.core.model import Label
from echora.classify.port import SpanDecision, SpanQuery
from echora.classify.prompt import (
    build_user_message, parse_response, response_schema, system_prompt,
)

# Measured best on the gold set: 95.2% overall / 92.5% collision.
# NB "llama-3.3-70b-versatile" is NOT a live Groq model id -- verify against
# client.models.list() before changing this.
DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqSpanClassifier:
    """Classifies spans with a Groq-hosted model.

    Requests strict JSON-Schema-constrained output, falling back to loose JSON
    object mode if the selected model does not support the strict mode -- the
    lenient parser in :mod:`echora.classify.prompt` copes with either.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = 10.0,
        strict_schema: bool = True,
    ) -> None:
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "GroqSpanClassifier needs the 'groq' package: "
                "pip install 'echora[groq]'"
            ) from exc

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. Export it, put it in .env, or "
                "configure a different classifier."
            )
        self._client = Groq(api_key=key, timeout=timeout)
        self._model = model
        self._strict = strict_schema

    def _response_format(self, spans: list[SpanQuery]) -> dict:
        if not self._strict:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "span_labels",
                "strict": True,
                "schema": response_schema(spans),
            },
        }

    def classify(self, sentence: str, spans: list[SpanQuery]) -> list[SpanDecision]:
        if not spans:
            return []

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt(spans)},
                {"role": "user", "content": build_user_message(sentence, spans)},
            ],
            response_format=self._response_format(spans),
            temperature=0,
            max_tokens=512,
        )

        parsed = parse_response(completion.choices[0].message.content, spans)
        return [
            SpanDecision(
                s.span_id,
                Label(parsed[s.span_id][0]) if s.span_id in parsed else Label.EN,
                parsed.get(s.span_id, (None, "unsure", None))[1],
                parsed.get(s.span_id, (None, None, None))[2],
            )
            for s in spans
        ]
