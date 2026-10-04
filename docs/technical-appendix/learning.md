# Technical lessons and open questions

These are conclusions supported by repository evidence, recorded retrospectively on 28 September 2026.

## What changed in the technical understanding

**A better aggregate command score does not guarantee benefit to the intended speaker.** Command-v3 improved the controlled-command benchmark while regressing on three personal clips. Per-speaker failures belong beside headline metrics.

**Beam coverage and reliable selection are different problems.** The reference may occur in a beam set without a reliable way to choose it, and sometimes every beam is wrong. Preserve alternatives and evaluate an acceptance policy separately from oracle coverage.

**Zero observed harm can result from an inactive mechanism.** Context had zero fitted weight and the learned acceptance gate was disabled. Zero harmful flips and zero automatic false accepts therefore cannot establish that those mechanisms are effective.

**Speech control is a state-management problem as well as a model problem.** A good candidate is insufficient if a delayed response can speak after Stop or an edit. Revision-bound authorization makes this requirement testable across clients.

## Next questions

1. Can a new independent speaker/phrase evaluation establish a useful verifier acceptance rate at the predeclared error bound?
2. Which failures arise from missing beam coverage versus incorrect ranking, and which adaptation changes improve the former without harming personal/normal speech?
3. Does active context improve intended-message selection when contradiction controls pass and weighting is nonzero?
4. Do blinded human ratings agree with the bounded wording grader, particularly for negation, names and multilingual outputs?
5. What effort and error rates do real speakers experience when choosing alternatives, editing and stopping speech?
6. Can the current model and frontend memory/latency costs be reduced while preserving these contracts?

These questions require new evidence. They are not completed features or scheduled experiments.
