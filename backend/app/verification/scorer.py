"""A small sequence-sensitive audio/text cross-encoder on frozen Qwen frames.

The scorer sees neither beam likelihoods nor personal context. Its output is a
match logit, not a probability of a correct transcription.
"""
from __future__ import annotations

import math
import re
import unicodedata

import torch
from torch import nn


FORMAT = "echora-acoustic-verifier-v1"
FEATURE_VERSION = "command-v3-projected-audio-2048-v1"
DEFAULT_CONFIG = {"input_size": 2048, "dimension": 192, "heads": 4,
                  "layers": 2, "feedforward": 384, "dropout": 0.1}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold().replace("’", "'")
    return " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, re.UNICODE))


def text_ids(text: str) -> list[int]:
    # Zero is reserved for padding. UTF-8 has no unknown-word token.
    return [value + 1 for value in normalize(text).encode("utf-8")]


def positions(length: int, dimension: int, device, dtype):
    index = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
                          * (-math.log(10000.0) / dimension))
    result = torch.zeros((length, dimension), device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(index * frequency)
    result[:, 1::2] = torch.cos(index * frequency)
    return result.to(dtype)


class AcousticTextScorer(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        dim = int(self.config["dimension"])
        if dim % 2 or dim % int(self.config["heads"]):
            raise ValueError("Scorer dimension must divide evenly into attention heads")
        self.audio = nn.Sequential(nn.LayerNorm(int(self.config["input_size"])),
                                   nn.Linear(int(self.config["input_size"]), dim))
        self.text = nn.Embedding(257, dim, padding_idx=0)
        self.segment = nn.Embedding(3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=int(self.config["heads"]),
            dim_feedforward=int(self.config["feedforward"]),
            dropout=float(self.config["dropout"]), batch_first=True, norm_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, int(self.config["layers"]),
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2),
                                  nn.GELU(), nn.Linear(dim // 2, 1))

    def forward(self, features: torch.Tensor, texts: list[str]) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.config["input_size"]:
            raise ValueError("Expected an untruncated [time,2048] Qwen acoustic sequence")
        if not features.shape[0] or not torch.isfinite(features).all():
            raise ValueError("Audio features must be nonempty and finite")
        if not texts:
            return features.new_empty((0,))
        encoded = [text_ids(text) for text in texts]
        if any(not value for value in encoded):
            raise ValueError("An acoustic candidate must contain literal text")
        device, dtype = self.cls.device, self.cls.dtype
        frames = self.audio(features.to(device=device, dtype=dtype))
        dim = frames.shape[-1]
        frames = frames + positions(len(frames), dim, device, dtype) + self.segment.weight[1]
        longest = max(map(len, encoded))
        tokens = torch.zeros((len(texts), longest), device=device, dtype=torch.long)
        for index, value in enumerate(encoded):
            tokens[index, :len(value)] = torch.tensor(value, device=device)
        wording = self.text(tokens) + positions(longest, dim, device, dtype) + self.segment.weight[2]
        sequence = torch.cat((self.cls.expand(len(texts), -1, -1) + self.segment.weight[0],
                              frames.unsqueeze(0).expand(len(texts), -1, -1), wording), dim=1)
        padding = torch.cat((torch.zeros((len(texts), len(frames) + 1), device=device,
                                         dtype=torch.bool), tokens == 0), dim=1)
        return self.head(self.encoder(sequence, src_key_padding_mask=padding)[:, 0]).squeeze(-1)


def contrastive_loss(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """Listwise matching plus class-balanced absolute matching; supports ties."""
    positive = positive.to(device=logits.device, dtype=torch.bool)
    if not positive.any() or positive.all():
        raise ValueError("Contrastive examples need both matching and mismatching pairs")
    listwise = torch.logsumexp(logits, 0) - torch.logsumexp(logits[positive], 0)
    binary = (torch.nn.functional.softplus(-logits[positive]).mean()
              + torch.nn.functional.softplus(logits[~positive]).mean()) / 2
    return (listwise + binary) / 2
