# ADR 0001: Preserve literal evidence through composition

Status: implemented; recorded retrospectively on 28 September 2026.

## Context and alternatives

The top transcript can be wrong, and a fluent rewrite can conceal that error. Options were top-one transcription only, unconstrained rewriting, or multiple immutable literal hypotheses followed by bounded composition.

## Decision and rationale

Retain all distinct literal beams and original scores. Use one grounded composition pass for adapted recognition, keeping generated messages, source IDs and literal alternatives separate. Relative search weights are not confidence.

## Trade-offs

The user can inspect alternative words and choose a literal fallback, while constrained composition limits unsupported additions. More visible choices increase interaction effort. Lexical grounding cannot prove general semantic fidelity or recover absent words.

## Evidence and consequences

The saved [I water failure](../../models/echora-qwen3-asr-command-v3/evaluation/next_decision.json) shows the reference absent from all five beams. The [recognition bridge](../../communication/backend/recognition.py), [grounding chain](../../backend/app/messaging/groq_chain.py) and [fidelity tests](../../backend/tests/test_unified_fidelity.py) implement and exercise the boundary.

Revisit when matched intended-message evaluations show an alternative preserves speaker control and meaning with less effort.
