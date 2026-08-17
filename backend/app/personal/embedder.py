"""A small local sentence encoder, loaded straight from `transformers`.

`sentence-transformers` is deliberately not a dependency here. This repo pins
`transformers==5.15.0` and that library has historically pinned itself below
transformers 5, so adding it risks dragging the recognizer's own stack backwards.
For this checkpoint it is also unnecessary: MiniLM's published pipeline is a
transformer, mean pooling and L2 normalization, which is what runs below.

Mean pooling is not an implementation detail. The retrieval query is a beam set
rather than a sentence, and each word carries the share of the search weight its
position gave it -- so the pooling step has to accept per-word weights. A
CLS-pooled encoder such as BGE could not express that at all.

CPU on purpose. The 1.7B recognizer owns MPS during a request, and 22M
parameters over a short string is a few milliseconds on CPU; queueing behind the
recognizer would add latency and save nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class Embedder:
    def __init__(self, root: Path, device: str = "cpu", max_tokens: int = 128) -> None:
        encoder = Path(root) / "encoder"
        if not encoder.is_dir():
            raise RuntimeError(
                f"No embedding bundle at {encoder}. Run scripts/fetch_embedder.sh, "
                "or set ECHORA_PERSONAL_ENABLED=false to run without personal context."
            )
        self.root = Path(root)
        self.device = device
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(str(encoder), local_files_only=True)
        if not self.tokenizer.is_fast:
            # word_ids() is what maps sub-tokens back to words. Without it the
            # weighted query would silently become an unweighted one, which is a
            # quiet accuracy loss rather than a visible failure.
            raise RuntimeError("The embedding bundle needs a fast tokenizer (tokenizer.json)")
        self.model = AutoModel.from_pretrained(str(encoder), local_files_only=True)
        self.model.eval()
        self.model.to(device)
        self.dimension = int(self.model.config.hidden_size)
        manifest = self.root / "manifest.json"
        if manifest.is_file():
            declared = json.loads(manifest.read_text(encoding="utf-8")).get("dimension")
            if declared and int(declared) != self.dimension:
                raise RuntimeError(
                    f"Embedding bundle declares dimension {declared} but the weights are {self.dimension}"
                )

    def _pool(self, text: str, weights: dict[str, float] | None) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )
        mask = encoded["attention_mask"].to(torch.float32)
        if weights:
            words = text.split()
            scale = torch.ones_like(mask)
            for position, index in enumerate(encoded.word_ids(0) or []):
                if index is None or index >= len(words):
                    continue
                scale[0, position] = float(weights.get(words[index].lower(), 1.0))
            mask = mask * scale
        with torch.inference_mode():
            hidden = self.model(**{key: value.to(self.device) for key, value in encoded.items()}).last_hidden_state
        mask = mask.to(hidden.device).unsqueeze(-1)
        total = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (hidden * mask).sum(dim=1) / total
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled[0].to(torch.float32).cpu().numpy()

    def encode(self, text: str, weights: dict[str, float] | None = None) -> np.ndarray:
        return self._pool(text.strip() or " ", weights)

    def encode_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self.encode(text) for text in texts]).astype(np.float32)
