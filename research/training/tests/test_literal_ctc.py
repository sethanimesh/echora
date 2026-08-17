import torch

from research.training.literal_ctc import (
    CTC_BLANK_ID,
    CTC_TO_ID,
    LiteralCTCHead,
    ctc_loss,
    decode_ids,
    encode_literal,
)


def test_literal_codec_preserves_telegraphic_phrase():
    encoded = encode_literal("I water!")
    expanded = []
    for item in encoded:
        expanded.extend([item, item, CTC_BLANK_ID])
    assert decode_ids(expanded) == "i water"


def test_ctc_rejects_alignment_without_repeat_frames():
    # "letter" has a repeated t and needs one more frame than characters.
    logits = torch.zeros((6, len(CTC_TO_ID)), requires_grad=True)
    loss, input_length, target_length = ctc_loss(
        logits, "letter", torch.nn.CTCLoss(blank=0)
    )
    assert loss is None
    assert (input_length, target_length) == (6, 6)


def test_ctc_head_and_loss_backpropagate():
    head = LiteralCTCHead(input_size=4)
    logits = head(torch.randn(20, 4))
    loss, _, _ = ctc_loss(logits, "i water", torch.nn.CTCLoss(blank=0))
    assert loss is not None
    loss.backward()
    assert head.projection.weight.grad is not None
