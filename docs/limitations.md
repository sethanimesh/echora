# Limitations and Threats to Validity

## Population and Task

- Eight dysarthric speakers across the folds cannot represent stroke survivors' diverse communication needs.
- Composed commands test sequencing, gaps and stretching; they are not natural commands or additional speakers.
- No participant study establishes usability, reduced communication effort, clinical benefit, or gaze/camera accessibility suitability.

## Experimental Validity

- The foundation gate is a selection screen, not an independent final evaluation.
- M04 was originally protected for command-v3, then historically exposed. In the later verification study, 227 of 229 M04 prompt groups also occur in training, covering 279 of 281 recordings.
- F03 had prior model-selection use. Later development results are not an untouched replication.
- Context weight was zero. Ablations cannot establish the benefit or safety of active personalized reranking.

## Measurement

- Literal WER is not intended-message accuracy. Oracle coverage is not delivered performance.
- Beam weights are relative search scores, not calibrated confidence.
- Bounded lexical checks and a related automated grader cannot prove general semantic fidelity. Translation lacks the same English guard.
- Reference edit distance is not measured user effort. Human wording ratings remain pending.
- Shared-Mac offline timings are not microphone-to-speech latency or native-device responsiveness.

## Reproduction and Deployment

Weights, corpora and later detailed outputs are excluded. Some historical recipes retain original machine paths. The supported application environment is a local Mac; this is not a hardened multi-user service. Provider outputs can change.

Automated checks establish controlled program behavior, not live microphone, audible playback, provider quality, or camera/gaze accuracy. [Reproducibility](reproducibility.md) specifies what a fresh clone can verify.