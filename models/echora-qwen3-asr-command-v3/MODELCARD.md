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
---

# Echora Qwen3-ASR command-v3

An adapter over `Qwen/Qwen3-ASR-1.7B-hf` for **literal transcription of dysarthric speech**.
It is one component of Echora, a communication assistant for stroke survivors.

This adapter emits up to **five literal hypotheses** with beam-search scores. It does not
decide what the speaker meant. Intent grouping, ranking, and grammar repair happen in the
application layer and are forbidden from overwriting the literal hypotheses. Raw ASR is
treated as immutable evidence so a speaker can always see the words the model actually
heard rather than a cleaned-up guess presented as fact.

This repository contains the adapter only. The Qwen foundation is **pinned, not copied** —
see Reproducibility below.

## Contents

| File | Size | Purpose |
| --- | ---: | --- |
| `adapter.safetensors` | 206 MB | Tuned audio layers, multimodal projector, LoRA state |
| `adapter_config.json` | 830 B | Artifact identity, literal prompt, hashes |

## Intended use

Assistive communication with visible alternatives and user control. In the current app, a resolved completed suggestion may speak on arrival; choosing an alternative speaks the selected words. Confirmation is internal revision-bound authorization, not an additional user step. The local learned verification route currently requests a choice because its acceptance audit failed. See [current architecture](../../docs/architecture.md).

## Out of scope

- **Not for clinical or diagnostic use.** No IRB, no clinical population study, and no
  assessment by a speech-language pathologist was performed.
- **Not general-purpose ASR.** The normal-speech number below is a retention guard, not a
  competitive claim.
- **Not for unattended transcription.** Nothing here is safe to use where no human confirms
  the output.

## Evaluation

Speaker-disjoint folds. **M04 is a protected outer test speaker**, sealed during training and
model selection. `v1` is the previous Echora adapter and the stated baseline. All numbers are
literal ASR, before any ranking, semantic repair, or personal context.

### Protected test — word error rate, lower is better

| Test set | v1 | v3 | Change |
| --- | ---: | ---: | ---: |
| TORGO (M04, sealed speaker) | 56.29% | **55.02%** | −1.27 pts |
| Composed commands | 72.58% | **51.58%** | −21.00 pts |
| Normal speech (retention guard) | 5.23% | **5.23%** | unchanged |
| Personal clips (3 utterances) | 22.22% | **44.44%** | **+22.22 pts — worse** |

### Top-5 beam diagnostics — 480 command utterances

| Measure | v1 | v3 | Change |
| --- | ---: | ---: | ---: |
| Rank-1 WER | 77.58% | **61.75%** | −15.83 pts |
| Rank-1 CER | 55.36% | **44.40%** | −10.96 pts |
| Top-5 oracle WER | 60.00% | **46.83%** | −13.17 pts |
| Exact literal present in top 5 | 16.04% | **36.46%** | +20.42 pts |

The top-5 oracle is diagnostic only and is never substituted for rank-1 WER.

### Known regression

**On the three personal clips, v3 is worse than v1** — 44.44% against 22.22% WER.

Those clips never entered training or model selection, and three utterances cannot outweigh a
480-utterance protected test, so v3 was still the correct deployment choice
(`deploy_v3_literal_multi_hypothesis`, all three test guards passed). But the deployed model
regressed on the exact speaker the project exists to serve. Anyone reading only the
command-set result is not seeing this model.

### Known failure: `I water`

Neither adapter recovers this reference phrase. It is grammatically incomplete, which is
precisely the kind of telegraphic speech the system is built for.

| Rank | v1 | v3 |
| ---: | --- | --- |
| 1 | gotten | thigh button |
| 2 | garden | thigh gotten |
| 3 | high, bottom | high button |
| 4 | high button | nine button |
| 5 | bottle | thigh bottom |

`literal_present: false` in both. v3's beams are more consistently two-word — the composition
training doing its job on structure — but the words are still wrong. This is the clearest
argument for keeping every alternative visible when selection is uncertain: collapsing to
rank 1 here would confidently present `thigh button` as what the speaker said.

## Scores are not confidence

Beam weights are **relative search scores within one beam set**. They are not calibrated
probabilities and must never be displayed as certainty percentages.

## Training data

| Split | Speakers | Commands | Hours | 2-word / 3-word |
| --- | --- | ---: | ---: | ---: |
| Train | F01, F04, M01, M02, M03, M05 | 6,000 | 14.6240 | 3,000 / 3,000 |
| Development | F03 | 480 | 0.9643 | 240 / 240 |
| Protected test | M04 | 480 | 1.2331 | 240 / 240 |

Sources are TORGO dysarthric recordings, a pool of composed commands built from real TORGO
isolated-word recordings, and Common Voice for the normal-speech retention pool. Splits have
equal coverage of four gap classes (to 3.2 s) and four stretch classes (to 2×), with no
speaker or composed-phrase overlap.

**The composed commands are controlled augmentation, not naturally spoken commands.** They
train and test literal sequencing, pauses, and stretch. They do not increase the number of
dysarthric speakers, which remains **eight** across these folds — the real ceiling on how far
these results generalise.

## Training configuration

- Tuned: audio layers 20–23, multimodal projector, rank-8 LoRA (α 16, dropout 0.05) on
  `q_proj`/`v_proj` in decoder layers 20–27
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

The literal instruction is **part of the contract, not a suggestion** — decoding without it
changes the behaviour these numbers describe:

> Transcribe only the words actually spoken. Keep incomplete or unusual word sequences
> literal. Do not add missing words.

Decode with 5 beams and return all hypotheses with their sequence scores.

## Ethical considerations

Trained on dysarthric speech from a research corpus. Recordings and local personal stores are excluded from this source repository.

Misrecognition in assistive communication carries real cost. A wrong word presented
confidently is worse than a visible uncertainty — which is why this model returns hypotheses
rather than an answer. The current application preserves alternatives and revision-bound speech authorization; see the interaction policy above.

## Limitations, in short

- TORGO has only eight dysarthric speakers in these train/dev/test folds.
- Command compositions are controlled augmentations, not naturally spoken commands.
- Beam probabilities are relative search scores, not calibrated confidence.
- No semantic repair, candidate ranker, or personal context was used in any result above.
- v3 regressed against v1 on the three personal clips.
- These are research results on protected test folds, not universal-ASR claims.
