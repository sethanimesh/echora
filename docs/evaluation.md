# Evaluation and evidence

These protocols are distinct. Different decoding settings, datasets and application boundaries prevent treating them as one directly comparable series.

## Foundation selection

The fixed screen contains 400 utterances, 50 per dysarthric speaker. Its primary metric is speaker-macro WER, with deletion and empty-output guards.

| Foundation | Speaker-macro WER | Deletion rate | Empty output |
| --- | ---: | ---: | ---: |
| Parakeet-TDT 1.1B | 45.83% | 5.84% | 1.25% |
| Qwen3-ASR 1.7B | 41.90% | 2.30% | 0.00% |

Sources: [comparison](../research/benchmarks/results/foundation_gate/comparison.json), [Parakeet predictions](../research/benchmarks/results/foundation_gate/parakeet_tdt_1_1b.tsv), [Qwen predictions](../research/benchmarks/results/foundation_gate/qwen3_asr_1_7b.tsv). `python3 scripts/reproduce_results.py` recomputes the aggregates. This is a selection screen, not untouched final validation.

## Command-v3 adaptation

Selected epoch: seven. Training updates acoustic layers, the multimodal projector, and decoder LoRA from the pinned foundation. F03 supplies development data, M04 the original protected speaker test. Composed commands use 6,000 training and 480 development/test examples each, with speaker- and composed-phrase-disjoint partitions.

| Saved protected-test metric | Previous v1 adapter | Command-v3 |
| --- | ---: | ---: |
| Original TORGO WER | 56.29% | 55.02% |
| Composed-command WER | 72.58% | 51.58% |
| Normal-speech WER | 5.23% | 5.23% |
| Personal-recording WER (three utterances) | 22.22% | 44.44% |

The separate five-beam command diagnostic reports **61.75% rank-one WER**, **46.83% oracle WER**, and **36.46% exact-in-top-five coverage** for v3. Do not substitute oracle coverage for delivered accuracy or conflate the diagnostic rank-one result with the protected-test result above.

Sources: [deployment decision](../models/echora-qwen3-asr-command-v3/evaluation/next_decision.json), [full result](../models/echora-qwen3-asr-command-v3/evaluation/result.json), [configuration](../models/echora-qwen3-asr-command-v3/evaluation/qwen_command_v3.json), [model report](../models/echora-qwen3-asr-command-v3/MODELCARD.md). The previous adapter is the baseline here, not Parakeet or unadapted Qwen.

## Later verifier and wording experiments

The [28 September pipeline report](pipeline-evaluation.md) records a separate production five-beam experiment. On 281 M04 recordings, ASR WER was 52.62% and fusion 52.33%; the paired difference was −0.28 percentage points (95% interval −1.02 to +0.50). Improvement is not established. Context fitting selected zero weight; zero harmful context flips do not establish safety of active context influence.

The acceptance audit accepted only 14 independent prompt groups against a requirement of 20. Its upper false-acceptance bound was 19.26% despite zero observed errors. The artifact remains advisory. All test cases requested clarification, so accepted-selection error is not estimable.

The wording experiment used 16 synthetic cases, two repeats and two conditions (64 trials). The bounded grader detected no meaning violations. Mean minimum reference word edits were 0.71875 without retrieved context and 0.81250 with it; this does not demonstrate a retrieval advantage. Human ratings remain pending.

Detailed later artifacts live in ignored `data/derived/verification/` and are **not distributed in a fresh clone**. The repository supports independent recomputation of foundation metrics and inspection of saved command-v3 summaries. It does not currently provide complete independent reanalysis of the later verifier study.

## Interpretation

The evidence demonstrates an inspectable evidence-to-message pipeline, a measured command-adaptation trade-off, preserved failures, and a gate that stays disabled after failing acceptance. It does not demonstrate universal dysarthric recognition, clinical benefit, an unseen-phrase advantage, or an improvement from personalized retrieval.

[Failures](failure-analysis.md) · [Validity limits](limitations.md) · [Reproduction](reproducibility.md)
