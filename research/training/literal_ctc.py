"""Small English character-CTC head for literal Qwen audio decoding."""

from __future__ import annotations

import torch

try:
    from metrics import normalize
except ImportError:  # imported as training.literal_ctc in local tests
    from research.benchmarks.metrics import normalize


CTC_SYMBOLS = ["<blank>", "<unk>", " ", "'", *list("abcdefghijklmnopqrstuvwxyz0123456789")]
CTC_TO_ID = {symbol: index for index, symbol in enumerate(CTC_SYMBOLS)}
CTC_BLANK_ID = CTC_TO_ID["<blank>"]
CTC_UNK_ID = CTC_TO_ID["<unk>"]


def normalize_literal(text: str) -> str:
    return normalize(text)


def encode_literal(text: str) -> list[int]:
    return [CTC_TO_ID.get(character, CTC_UNK_ID) for character in normalize_literal(text)]


def collapse_ids(ids: list[int]) -> list[int]:
    collapsed = []
    previous = None
    for index in ids:
        if index != previous and index != CTC_BLANK_ID:
            collapsed.append(index)
        previous = index
    return collapsed


def decode_ids(ids: list[int]) -> str:
    characters = []
    for index in collapse_ids(ids):
        symbol = CTC_SYMBOLS[index]
        characters.append("?" if symbol == "<unk>" else symbol)
    return " ".join("".join(characters).split())


class LiteralCTCHead(torch.nn.Module):
    def __init__(self, input_size: int = 2048):
        super().__init__()
        self.projection = torch.nn.Linear(input_size, len(CTC_SYMBOLS))

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.projection(audio_features)


def ctc_loss(
    logits: torch.Tensor,
    transcript: str,
    loss_function: torch.nn.CTCLoss,
) -> tuple[torch.Tensor | None, int, int]:
    if logits.ndim != 2:
        raise ValueError(f"Expected packed [time, classes] CTC logits, got {tuple(logits.shape)}")
    target_ids = encode_literal(transcript)
    input_length = logits.shape[0]
    target_length = len(target_ids)
    # CTC needs an extra frame between identical adjacent labels. Checking only
    # target_length <= input_length can otherwise produce an infinite loss for
    # words such as "letter" even though the raw lengths appear valid.
    repeated_labels = sum(
        left == right for left, right in zip(target_ids, target_ids[1:])
    )
    minimum_input_length = target_length + repeated_labels
    if not target_ids or minimum_input_length > input_length:
        return None, input_length, target_length
    targets = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
    log_probs = logits.log_softmax(dim=-1).unsqueeze(1)
    input_lengths = torch.tensor([input_length], dtype=torch.long, device=logits.device)
    target_lengths = torch.tensor([target_length], dtype=torch.long, device=logits.device)
    loss = loss_function(log_probs, targets, input_lengths, target_lengths)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite CTC loss for target length {target_length}")
    return loss, input_length, target_length


def greedy_decode(logits: torch.Tensor) -> str:
    if logits.ndim != 2:
        raise ValueError(f"Expected [time, classes], got {tuple(logits.shape)}")
    return decode_ids(logits.argmax(dim=-1).detach().cpu().tolist())
