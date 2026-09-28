# ADR 0002: Separate verification from generated-candidate count

Status: implemented; learned automatic acceptance disabled. Recorded retrospectively on 28 September 2026.

## Context and alternatives

One surviving generated option does not imply unambiguous acoustics. Alternatives were the legacy candidate-count rule, ASR rank one alone, and a separate audio–text verifier with calibration and acceptance gates.

## Decision and rationale

For local adapted English recognition, score only frozen Qwen features and literal text. Apply profile context afterward through bounded fusion. Require artifact identity and calibration/acceptance checks before learned automatic selection. Missing or failed artifacts request a choice. Other routes retain explicitly labelled legacy behavior.

## Trade-offs and evidence

This creates an auditable decision boundary but requires representative independent calibration data and can increase clarification. The [later acceptance audit](../pipeline-evaluation.md) failed: only 14 accepted prompt groups against a minimum of 20. The implementation therefore leaves the artifact advisory. The WER interval crosses zero and learned context weight is zero; neither acoustic nor context benefit is established.

See [verification implementation](../../backend/app/verification/) and [tests](../../backend/tests/test_verification.py). Revisit with new independent calibration/acceptance data under a prospectively frozen protocol, not by adjusting thresholds after inspecting the test result.
