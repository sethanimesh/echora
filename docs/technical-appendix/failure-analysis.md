# Failure analysis

## Correct phrase absent from every beam

For the personal reference `I water`, command-v3 returns `thigh button`, `thigh gotten`, `high button`, `nine button`, and `thigh bottom`. Neither adapter contains the literal reference. See [`personal_i_water` in the saved decision](../models/echora-qwen3-asr-command-v3/evaluation/next_decision.json).

The recognizer did not recover the phrase; the evidence does not isolate one acoustic cause. A ranker cannot select an absent candidate. Retaining alternatives, typed edits and repeat recording preserves control but does not solve recognition. In the later M04 experiment, all beams were wrong on 120 of 281 recordings.

## Aggregate gain hides a personal regression

Command-v3 improves the composed-command test but raises WER from 22.22% to 44.44% on three personal recordings. The sample is small, but the observed regression is real. Aggregate gains do not establish benefit for that speaker. [Saved result](../models/echora-qwen3-asr-command-v3/evaluation/next_decision.json).

## Insufficient acceptance evidence

The learned audit reached 14 accepted prompt groups against a minimum of 20. Zero observed errors still permits a 19.26% upper false-acceptance bound. The application keeps the artifact advisory, producing clarification rather than demonstrating successful automatic selection. [Pipeline report](pipeline-evaluation.md).

## Contradiction filter misses double negation

The later control audit retrieved opposing `no` / `not no` fixtures in five M04 candidate slots. Double negation and an empty base escape the lexical filter. Context weight was zero, so these references did not change ranking, but the retrieval control still failed. The frozen experiment was not retuned after inspecting M04. [Audit description](pipeline-evaluation.md).

## Late work outlives a user action

A stopped model call may continue computing. Revision authorization, cancellation checks and model locking prevent its late output from becoming current speech or overlapping another model call. Native cross-session revocation uses polling, subject to network delay. This is a tested failure scenario, not complete real-device timing validation. [Lifecycle](architecture.md#message-lifecycle-and-speech).
