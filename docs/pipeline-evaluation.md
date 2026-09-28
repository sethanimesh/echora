# Integrated pipeline evaluation — 28 September 2026

The application integration and the learned experiment are separate deliverables. The code preserves five-beam literal evidence, ranks with bounded acoustic/context components, retains explicit alternatives, and supports profile-scoped, explicitly remembered wording. A trained artifact may authorize automatic selection only after its declared calibration and untouched acceptance gates pass.

## Completed wording experiment

The full production `recognition.compose` boundary completed 64 synthetic English trials: 16 frozen cases, two repeats, and plain/retrieved conditions. Every condition produced 32 suggestions. The bounded grader found zero displayed or provider-output meaning violations, provider failures, abstentions or withheld outputs in the completed run.

| Condition | Suggestions | Detected meaning violations | Mean minimum reference word edits |
| --- | ---: | ---: | ---: |
| Plain | 32 | 0 | 0.71875 |
| Retrieved context | 32 | 0 | 0.81250 |

These results do not demonstrate a correction advantage from retrieved context. Edit distances measure differences from approved reference examples, not actual user effort. The deterministic grader shares bounded checks with production and cannot prove unrestricted intent preservation. The 64-row blinded pack awaits human ratings.

Local artifacts:

- [Report](../data/derived/verification/expansion-v2/report.md)
- [Frozen manifest](../data/derived/verification/expansion-v2/manifest.json) and [inputs](../data/derived/verification/expansion-v2/frozen_inputs.json)
- [Blinded review](../data/derived/verification/expansion-v2/blinded_review.csv), [instructions](../data/derived/verification/expansion-v2/review_instructions.md), and [separate answer key](../data/derived/verification/expansion-v2/review_key.csv)
- [Provenance](../data/derived/verification/expansion-v2/provenance.md)

The initial four restricted-network failures remain archived separately. The earlier v1 chain-only results are preserved, including their capitalization-sensitive grading defect. V1 and v2 differ in application boundary and valid fixture construction; they are not a paired before/after model comparison. No real profile database or TORGO labels entered this wording experiment. Synthetic wording inputs were sent to the configured Groq provider.

## Completed acoustic experiment

All 4,170 fitting recordings were prepared locally. Training completed after 14 epochs under the frozen early-stopping rule, selecting epoch eleven. The separate scorer has 1,060,673 parameters and took about 80 minutes of optimizer and checkpoint-selection work, excluding audio preparation and negative generation. Its scorer-only WER on the 185-record F03 checkpoint-selection partition was 24.27%, versus 11.83% for ASR rank one. The scorer alone did not improve that development baseline. These are development measurements, not M04 results.

Fusion and calibration are frozen under artifact `98f6defc3d82864142250f3c`. Calibration covers 65 correct and 15 incorrect prompt groups, including seven all-beams-wrong groups; class-group counts can overlap across repeated recordings or context variants. Both integrated and context-only fitting selected zero contextual weight. The integrated acoustic gate still learns a small, adaptive scorer contribution; its saved static ASR prior is not the final per-record weight.

With zero contextual weight, full-fusion rankings equal acoustic reranking and context-assisted rankings equal ASR rank one. Zero harmful context flips in this run therefore cannot establish the safety or benefit of active context influence. Similarly, the failed audit disabled every learned automatic acceptance during testing; zero automatic false accepts under that policy do not demonstrate accurate automatic decisions.

The untouched acceptance audit **failed**: 14 independent prompt groups were accepted with zero incorrect accepted groups and zero harmful context flips, below the required 20 accepted groups. Its one-sided 95% upper false-acceptance bound is 19.26%, despite zero observed errors. The fixed threshold remains 0.95. The application successfully loaded the artifact in advisory mode and returned clarification with no automatically selected hypothesis. Learned automatic selection remains disabled.

All 281 original M04 recordings and 262 normal-speech test recordings were decoded once using the production prompt and five-beam settings. All four arms reused those literal outputs. The table applies to every frozen condition: empty, relevant, misleading, contradictory, stale and wrong recipient. These are controlled fixtures derived from training phrases, not participant histories; the relevant/misleading pool labels do not guarantee relevance for each individual utterance.

| M04 configuration | WER | CER | Exact match |
| --- | ---: | ---: | ---: |
| ASR rank one | 52.62% | 36.87% | 44.48% |
| Acoustic reranking (ASR + scorer) | 52.33% | 36.67% | 44.48% |
| ASR + context, without scorer | 52.62% | 36.87% | 44.48% |
| Full fusion | 52.33% | 36.67% | 44.48% |

M04's paired WER difference for fusion versus ASR was **−0.28 percentage points**, with a 95% interval of **−1.02 to +0.50 points**. CER changed by −0.20 points (95% interval −0.74 to +0.29), and exact match did not change. These intervals use 2,000 paired bootstrap replicates over 229 normalized prompt groups. This run does not establish a transcription improvement. Five-beam oracle coverage was 57.30% exact match and 41.73% WER; all beams were wrong on 120 of 281 recordings.

On the 262-record normal-speech holdout, ASR WER was 5.11%, versus 5.08% for acoustic reranking/full fusion; the paired difference was −0.04 points (95% interval −0.22 to +0.11). CER was 1.86% versus 1.84%, and exact match 67.56% versus 67.94%. ASR + context equaled ASR. Five-beam exact oracle coverage was 82.06%; all beams were wrong on 47 recordings.

Every test condition had 100% clarification and zero learned automatic accepts because the acceptance gate was disabled. Accepted-selection error is therefore not estimable. There were zero harmful context flips and zero accepted all-beams-wrong examples, with the policy limitations above. No threshold or weights were changed after the audit or M04 evaluation.

The post-run control audit found a **failed contradiction-filter control**: five M04 candidate slots across five recordings returned ten reference occurrences from the opposing `no` / `not no` fixture pair. This double-negation/empty-base case escapes the bounded polarity filter. Its fusion contribution was zero because context weighting was disabled, but the retrieval filter itself did not meet this control. Empty, stale and wrong-recipient fixtures returned no sources, and normal-speech contradictory fixtures returned none. This remains a documented filter limitation; the frozen method was not retuned after inspecting M04. Exact affected recording/source IDs and all checks are in the [post-run audit](../data/derived/verification/run-v1/evaluation-audit.json).

M04 has historical test exposure and is not a newly untouched test. Of its 229 prompt groups, 227 also occur in the training partition, covering 279 of 281 recordings. This is a speaker holdout with extensive repeated-prompt overlap, not an unseen-phrase or population study. F03 was previously used in ASR model selection. The frozen experiment used 3,613 training recordings (2,813 TORGO, including a predeclared 800-control cap, plus 800 normal-speech recordings) and 557 F03 development recordings. Long files were wholly excluded under the predeclared 45-second scope; no M04 recording was excluded or truncated. Full eligibility and overlap counts are in the machine-readable results.

Across the 543 test recordings, median decoding plus feature extraction was 2.29 seconds (95th percentile 3.98 seconds); the separate CPU scorer added a median 2.27 ms (95th percentile 5.42 ms). Context-fixture evaluation took 10.30 seconds total, including MiniLM loading and cached fixture vectors across six conditions. These are offline sequential measurements on a shared Mac with other applications and a brief concurrent validation run, not microphone-to-speech latency or an isolated performance benchmark.

Local artifact files:

- [Readable acoustic report](../data/derived/verification/run-v1/report.md), [full matched results](../data/derived/verification/run-v1/evaluation.json), and [per-record condition results](../data/derived/verification/run-v1/evaluation.jsonl)
- [Trained scorer](../data/derived/verification/run-v1/artifact/scorer.safetensors), [artifact manifest](../data/derived/verification/run-v1/artifact/manifest.json), [fusion](../data/derived/verification/run-v1/artifact/fusion.json), and [calibration/audit](../data/derived/verification/run-v1/artifact/calibration.json)
- [Training history](../data/derived/verification/run-v1/training.json), [resumable checkpoint](../data/derived/verification/run-v1/checkpoint.pt), [frozen protocol](../data/derived/verification/run-v1/protocol.json), and [negative provenance](../data/derived/verification/run-v1/training-pairs.json)

Audio preparation resumed after an initial disk-full failure and later received unused-MPS-buffer cleanup; completed records were preserved. The saved twelve-recording benchmark measures extraction time. Memory telemetry was added to subsequent genuine cache records; those later observations must not be retroactively attributed to the initial benchmark. Training RSS high-water was about 624 MiB, and end-of-epoch MPS driver samples were about 327 MiB; GPU samples are not peak measurements. System swap caused temporary disk pressure during training, which recovered without changing the corpus or compute route.

During held-out Qwen recognition, maximum observed post-record samples were 3.797 GiB MPS live allocation, 4.283 GiB MPS driver memory and 628.41 MiB process RSS high-water. The full evaluation process reached 718.95 MiB RSS high-water. GPU samples follow buffer cleanup and do not measure transient peaks; the near-zero MPS allocation in the final report is after Qwen unloading. The post-run audit verified all 543 decodings were newly produced once and all 3,258 recording/condition rows retain identical cached evidence, candidate membership and failed-audit clarification.

## Verification evidence and limits

The final application checks completed:

| Check | Result |
| --- | --- |
| Isolated unified backend, language and native contracts | 708 passed, 2 skipped |
| Full research benchmark and training suites | 63 passed |
| Web tests | 77 passed |
| Native tests | 26 passed |
| Both client type checks | Passed |
| Web production build and iOS export | Passed |

The final combined Python run passed 771 tests with two skips. Lifecycle checks cover context-change revocation, selected-reading name preservation, retained fourth/fifth alternatives after manual edits and redrafts, missing-verification silence, cross-process deletion during generation, deleted-save revision revocation, repeated Stop while model inference is running, and recognition device/precision mismatch. The full research total includes the 17 expansion benchmark tests. These checks used dummy credentials and isolated storage; the Python runs explicitly blocked external connections. Existing framework/deprecation and large-client-bundle warnings did not prevent completion.

Deletion cancels pending derived wording and speech, and delayed responses cannot restore old authorization. Native cross-session revocation uses a 500 ms polling interval, subject to network delay. Source conflict checks are conservative lexical exclusions, not general semantic contradiction detection.

Automated checks do not establish microphone performance, physical playback quality, accessibility suitability or clinical benefit. See [implementation and resumable commands](verification-and-memory.md).
