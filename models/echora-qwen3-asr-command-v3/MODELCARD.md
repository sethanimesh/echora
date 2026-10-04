---
base_model: Qwen/Qwen3-ASR-1.7B-hf
pipeline_tag: automatic-speech-recognition
language:
  - en
tags:
  - dysarthria
  - dysarthric-speech
  - assistive-technology
  - aac
  - torgo
  - literal-asr
  - adapter
  - qwen3-asr
---

# Echora Qwen3-ASR Command-v3: Literal ASR Adapter for Composed Dysarthric Commands

An adapter over `Qwen/Qwen3-ASR-1.7B-hf` for **literal transcription of dysarthric speech on composed commands**. Emits up to 5 literal hypotheses with beam-search scores. Intent grouping, ranking, and grammar repair are application-layer concerns — raw ASR evidence remains immutable.

This repository contains the adapter only. The Qwen foundation is **pinned, not copied** — see Reproducibility below.

> `pipeline_tag: automatic-speech-recognition` — **Note:** Evaluated on composed dysarthric commands (8 speakers), not general ASR.

[Source code](https://github.com/sethanimesh/echora) · [Training protocol](https://github.com/sethanimesh/echora/blob/main/research/training/COMMAND-V3-README.md) · [Evaluation](https://github.com/sethanimesh/echora/blob/main/docs/evaluation.md) · [Architecture](https://github.com/sethanimesh/echora/blob/main/docs/architecture.md)

## Contents

| File | Size | Purpose |
| --- | ---: | --- |
| `adapter.safetensors` | 206 MB | Tuned audio layers, multimodal projector, LoRA state |
| `adapter_config.json` | 830 B | Artifact identity, literal prompt, hashes |

## Intended Use

**Literal transcription adapter for composed dysarthric commands.** This adapter emits up to 5 literal hypotheses with beam-search scores for downstream ranking and selection. The full Echora application includes an experimental learned verifier for acoustic reranking and context fusion (documented in the [Technical Appendix](https://github.com/sethanimesh/echora/blob/main/docs/technical-appendix/pipeline-evaluation.md)). This adapter provides the immutable literal evidence that feeds into that pipeline.

Suitable for research on dysarthric speech recognition, assistive communication prototypes, and literal ASR evidence preservation.

## Out of Scope

- **Not for clinical or diagnostic use.** No IRB, no clinical population study, and no assessment by a speech-language pathologist was performed.
- **Not general-purpose ASR.** The normal-speech number below is a retention guard, not a competitive claim.
- **Not a complete communication system.** Ranking, verification, and message composition are application-layer components.

## Highlighted Results

Speaker-disjoint folds. **M04 is a protected outer test speaker**, sealed during training and model selection. `v1` is the previous Echora adapter and the stated baseline. All numbers are literal ASR, before any ranking, semantic repair, or personal context.

### Protected Test (M04 — Speaker Holdout)

| Test Set | v1 | v3 | Change |
| --- | ---: | ---: | ---: |
| **Composed commands** | 72.58% | **51.58%** | **−21.00 pts** |
| Normal speech (retention guard) | 5.23% | **5.23%** | unchanged |
| TORGO (M04) | 56.29% | **55.02%** | −1.27 pts |

### Five-Beam Diagnostic (Separate Decode)

| Measure | v1 | v3 | Change |
| --- | ---: | ---: | ---: |
| Rank-1 WER | 77.58% | **61.75%** | −15.83 pts |
| Rank-1 CER | 55.36% | **44.40%** | −10.96 pts |
| **Top-5 oracle WER** | 60.00% | **46.83%** | −13.17 pts |
| **Exact literal in top 5** | 16.04% | **36.46%** | **+20.42 pts** |

> The top-5 oracle is diagnostic and demonstrates the value of preserving literal alternatives. The application layer handles selection.

### Key Finding: `I water` Reference

Neither adapter recovers this telegraphic reference phrase — it is absent from all 5 beams. This demonstrates why preserving every literal alternative is critical: collapsing to rank-1 would confidently present `thigh button` as the transcription. The 5-beam design ensures the speaker retains control.

| Rank | v1 | v3 |
| ---: | --- | --- |
| 1 | gotten | thigh button |
| 2 | garden | thigh gotten |
| 3 | high, bottom | high button |
| 4 | high button | nine button |
| 5 | bottle | thigh bottom |

## Scores Are Not Confidence

Beam weights are **relative search scores within one beam set**. They are not calibrated probabilities and must never be displayed as certainty percentages.

## Training Data

| Split | Speakers | Commands | Hours | 2-word / 3-word |
| --- | --- | ---: | ---: | ---: |
| Train | F01, F04, M01, M02, M03, M05 | 6,000 | 14.6240 | 3,000 / 3,000 |
| Development | F03 | 480 | 0.9643 | 240 / 240 |
| Protected test | M04 | 480 | 1.2331 | 240 / 240 |

Sources are TORGO dysarthric recordings, a pool of composed commands built from real TORGO isolated-word recordings, and Common Voice for the normal-speech retention pool. Splits have equal coverage of four gap classes (to 3.2 s) and four stretch classes (to 2×), with no speaker or composed-phrase overlap.

**The composed commands are controlled augmentation, not naturally spoken commands.** They train and test literal sequencing, pauses, and stretch. They do not increase the number of dysarthric speakers, which remains **eight** across these folds.

## Training Configuration

- Tuned: audio layers 20–23, multimodal projector, rank-8 LoRA (α 16, dropout 0.05) on `q_proj`/`v_proj` in decoder layers 20–27
- Trainable parameters: 53,992,448 of 2,038,511,232 (2.65%)
- Learning rates: 1.5e-5 audio, 8e-5 LoRA · weight decay 0.01 · grad accumulation 8
- Selected epoch 7 of 8 maximum · seed 20260818 · single NVIDIA L40S
- No candidate ranker, no semantic repair, no personal context, no LM rescoring

## Reproducibility

The foundation is pinned rather than redistributed:

```
Qwen/Qwen3-ASR-1.7B-hf
revision  bcd2b5b7f32b480ab5790554cfa8347f246a14f3
sha256    2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1
```

Adapter `sha256`: `7cd203cc0cbf479e6afa198cc3895fedc71f1907b870fcb7f1dece0a1e6b2021`

The literal instruction is **part of the contract, not a suggestion** — decoding without it changes the behaviour these numbers describe:

> Transcribe only the words actually spoken. Keep incomplete or unusual word sequences literal. Do not add missing words.

Decode with 5 beams and return all hypotheses with their sequence scores.

From a checkout of the [Echora source repository](https://github.com/sethanimesh/echora), download the released weights and configuration:

```python
from huggingface_hub import hf_hub_download

for filename in ("adapter.safetensors", "adapter_config.json"):
    hf_hub_download(
        repo_id="sethanimesh/echora-qwen3-asr-command",
        filename=filename,
        revision="v3",
        local_dir="models/echora-qwen3-asr-command-v3/adapter",
    )
```

Provision the pinned pretrained foundation following the [model-bundle guide](https://github.com/sethanimesh/echora/blob/main/docs/reproducibility.md#model-bundle). The supported application entry point is `./scripts/dev.sh`. Direct inference uses [QwenCommandEngine](https://github.com/sethanimesh/echora/blob/main/backend/app/asr/engine.py), supplied with the foundation directory, adapter file, and [inference configuration](https://github.com/sethanimesh/echora/blob/main/models/echora-qwen3-asr-command-v3/evaluation/qwen_command_v3.json).

## Ethical Considerations

Trained on identifiable dysarthric speakers from a licensed research corpus. Misrecognition in assistive communication carries real cost — which is why this model returns hypotheses rather than an answer, and why the surrounding application requires the speaker to confirm a message before it is spoken.

## Limitations

- TORGO has only **eight** dysarthric speakers in these train/dev/test folds.
- **Composed commands** are controlled augmentations, not naturally spoken commands.
- Beam probabilities are relative search scores, not calibrated confidence.
- No semantic repair, candidate ranker, or personal context was used in any result above.
- Speaker holdout M04 shares 227/229 prompt groups with training (279/281 recordings) — this is a speaker holdout with extensive prompt overlap, not an unseen-phrase or population study.
- These are research results on protected test folds, not universal-ASR claims.

[Failure analysis](https://github.com/sethanimesh/echora/blob/main/docs/technical-appendix/failure-analysis.md) and [pipeline evaluation](https://github.com/sethanimesh/echora/blob/main/docs/technical-appendix/pipeline-evaluation.md) describe the broader application boundaries.