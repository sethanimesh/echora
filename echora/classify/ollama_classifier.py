"""Local span classifier backed by Ollama. The first ``auto`` route.

It provides the best measured latency of the configured model routes and keeps
classification available when a network backend cannot be reached.
"""

from __future__ import annotations

from echora.core.model import Label
from echora.classify.port import SpanDecision, SpanQuery
from echora.classify.prompt import (
    build_user_message, parse_response, response_schema, system_prompt,
)

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_HOST = "http://localhost:11434"


class OllamaSpanClassifier:
    """Classifies spans with a locally-served model via Ollama.

    Uses Ollama's native ``format`` parameter (JSON Schema), which constrains
    decoding rather than merely requesting JSON -- the model cannot emit a label
    outside the enum.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 15.0,
        keep_alive: str = "30m",
        think: bool = False,
        max_batch: int = 0,
    ) -> None:
        # Imported lazily so `ollama` stays an optional extra: a deployment that
        # only uses Groq should not need it installed.
        try:
            from ollama import Client
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OllamaSpanClassifier needs the 'ollama' package: "
                "pip install 'echora[ollama]'"
            ) from exc

        self._client = Client(host=host, timeout=timeout)
        self._model = model
        # Keeps the model resident between utterances. Without this every
        # sentence pays a multi-second cold load, which is fatal for a
        # conversational device.
        self._keep_alive = keep_alive
        # Reasoning traces would burn hundreds of tokens before emitting one
        # label. Off by default; the task needs judgement, not deliberation.
        self._think = think
        # Spans per request; 0 means no cap. DEFAULT OFF, and the reason is
        # worth recording because the obvious conclusion is wrong.
        #
        # 4B models really do degrade by list position: "attend" in
        # "Main kal office mein meeting attend karunga" returns EN at index 0-1
        # and HI at index 2-3, deterministically (5/5 identical runs at
        # temperature 0). gemma3:4b shows it too; qwen3:8b does not, so it is a
        # small-model capacity limit. Rewriting the prompt does not help --
        # inline-marked spans and a repeated trailing sentence both still failed.
        # Capping the batch at 2 does fix that sentence completely.
        #
        # But measured on the full gold set it does NOT pay for itself:
        #   chunked (2)  89.3% overall / 97.5% collision
        #   unchunked    90.5% overall / 97.5% collision
        # Capping loses cross-span context, which costs more than the position
        # effect does -- once EN_ZIPF_DECISIVE removes the loanwords that
        # dominated long batches, most sentences carry too few ambiguous spans
        # to reach the degraded positions at all.
        #
        # Kept as an option, off by default: a deployment seeing long,
        # many-ambiguity sentences may still want max_batch=2. Re-measure before
        # turning it on.
        self._max_batch = max_batch if max_batch > 0 else None

    def warm_up(self) -> None:
        """Load the model before the user needs it.

        Measured warm latency is ~0.9-1.5 s, but the first request after the
        model is evicted pays a multi-second load. Calling this at startup moves
        that cost off the conversational path, where a stall reads as the device
        being broken.
        """
        try:
            self.classify("Main ghar ja raha hoon", [SpanQuery(0, "Main", 0, 4)])
        except Exception:  # noqa: BLE001 - warm-up is best-effort
            pass

    def classify(self, sentence: str, spans: list[SpanQuery]) -> list[SpanDecision]:
        if not spans:
            return []
        if self._max_batch and len(spans) > self._max_batch:
            out: list[SpanDecision] = []
            for start in range(0, len(spans), self._max_batch):
                out.extend(self._classify_chunk(spans[start:start + self._max_batch],
                                                sentence))
            return out
        return self._classify_chunk(spans, sentence)

    def _classify_chunk(
        self, spans: list[SpanQuery], sentence: str
    ) -> list[SpanDecision]:
        # Span ids are renumbered per chunk so the model always sees 0..n-1,
        # then mapped back, keeping the caller's ids stable.
        local = [
            SpanQuery(i, s.text, s.start, s.end, s.candidates)
            for i, s in enumerate(spans)
        ]
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt(local)},
                {"role": "user", "content": build_user_message(sentence, local)},
            ],
            format=response_schema(local),
            think=self._think,
            keep_alive=self._keep_alive,
            options={
                "temperature": 0,   # classification: no sampling diversity wanted
                "num_predict": 512,
            },
        )

        parsed = parse_response(response["message"]["content"], local)
        return [
            SpanDecision(
                original.span_id,
                Label(parsed[i][0]) if i in parsed else Label.EN,
                parsed.get(i, (None, "unsure", None))[1],
                parsed.get(i, (None, None, None))[2],
            )
            for i, original in enumerate(spans)
        ]
