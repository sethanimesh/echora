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

# Echora: Qwen3-ASR Adaptation for Dysarthric Speech

Echora is an English speech-recognition adapter for literal transcription of dysarthric speech and short communication commands. It adapts the pretrained Qwen3-ASR 1.7B foundation through selected acoustic layers, the multimodal projector, and decoder LoRA, updating **2.65% of the model's parameters**.

The released artifact supports up to five literal transcription hypotheses. It is integrated into a communication assistant that preserves recognition evidence while helping the speaker select and communicate a message.

[Source code](https://github.com/sethanimesh/echora) · [Training protocol](https://github.com/sethanimesh/echora/blob/main/research/training/COMMAND-V3-README.md) · [Evaluation](https://github.com/sethanimesh/echora/blob/main/docs/evaluation.md) · [Architecture](https://github.com/sethanimesh/echora/blob/main/docs/architecture.md)

## Model overview

| Property | Released model |
| --- | --- |
| Pretrained foundation | [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) |
| Task | English dysarthric speech and controlled short-command transcription |
| Adaptation | Acoustic-layer tuning, multimodal-projector tuning, decoder LoRA |
| Trainable parameters | 53,992,448 of 2,038,511,232 total (2.65%) |
| Adapter artifact | 215,982,640 bytes (216 MB / 206 MiB) |
| Output | Literal transcription; optional five-hypothesis decoding |
| Checkpoint selection | Epoch 7; development speaker F03 |

The adapter is downloadable from this model repository. The foundation is provisioned separately from its pinned upstream revision. The adaptation fraction describes training parameters; it does not imply reduced foundation inference memory.

## Evaluation against the pretrained foundation

The pretrained foundation and released adapter were evaluated on identical F03 development manifests using the same literal prompt and deterministic single-output decoding. F03 was used for checkpoint selection, so these are **development results**. Word error rate (WER) is lower-is-better; normalized exact match is higher-is-better.

| Development dataset | Recordings | Pretrained Qwen3-ASR WER | Echora WER | Relative WER reduction |
| --- | ---: | ---: | ---: | ---: |
| TORGO dysarthric speech, F03 | 557 | 22.16% | **11.51%** | **48.1%** |
| Controlled composed commands, F03 | 480 | 53.92% | **24.92%** | **53.8%** |

### Exact transcription and word preservation

| Development measure | Pretrained Qwen3-ASR | Echora |
| --- | ---: | ---: |
| TORGO normalized exact match | 56.37% | **76.30%** |
| Composed-command normalized exact match | 12.92% | **48.96%** |
| Composed-command word deletion rate | 6.17% | **0.50%** |

On the 480-command development split, deleted reference words decreased from **74 to 6** out of 1,200 reference words: a **91.9% relative reduction in word deletions**. This measures word omission under that protocol, rather than an overall accuracy or real-world communication benefit.

Source: [complete experiment result](https://github.com/sethanimesh/echora/blob/main/models/echora-qwen3-asr-command-v3/evaluation/result.json), fields `baseline.foundation_prompt` and `selected_development`. The same [evaluation implementation](https://github.com/sethanimesh/echora/blob/main/research/training/qwen_command_v3.py) measures both models. Normal-speech development measurements and all historical comparisons are retained in the detailed evaluation below.

## Held-out speaker evaluation

M04 was excluded from this adapter's training and checkpoint selection. Results are literal recognition, before wording generation, personal context, or acoustic reranking. WER and character error rate (CER) are lower-is-better.

| Test dataset | Recordings | WER | CER |
| --- | ---: | ---: | ---: |
| TORGO dysarthric speech, M04 | 281 | **55.02%** | **38.40%** |
| Controlled composed commands, M04 | 480 | **51.58%** | **30.52%** |
| Normal speech | 262 | **5.23%** | **1.86%** |

These figures characterize the released adapter. A matched pretrained-foundation comparison was not recorded for the command-v3 protected command test; development improvements above must not be extrapolated to that test.

### Multiple-hypothesis decoding

A separate five-beam diagnostic evaluated 480 composed commands:

| Diagnostic measure | Released adapter |
| --- | ---: |
| Rank-one WER | 61.75% |
| Rank-one CER | 44.40% |
| Top-five oracle WER | 46.83% |
| Exact reference present in the top five | 36.46% |

Oracle measures describe the best available hypothesis when the reference is known. They do not establish automatic-selection accuracy. This diagnostic uses a separate decoding protocol; its rank-one result is distinct from the single-output held-out table.

## Adaptation methodology

The experiment combines targeted acoustic adaptation with low-rank decoder updates:

- **Acoustic encoder:** tune layers 20–23.
- **Multimodal projector:** tune the audio-to-language projection.
- **Decoder:** rank-8 LoRA on `q_proj` and `v_proj` in layers 20–27; alpha 16; dropout 0.05.
- **Optimization:** audio learning rate 1.5e-5; decoder LoRA learning rate 8e-5; weight decay 0.01; gradient accumulation 8.
- **Training:** at most eight epochs; selected epoch seven; seed 20260818; one NVIDIA L40S.
- **Evidence boundary:** ASR evaluation contains no language-model repair, candidate ranking, or profile context.

Training scripts preserve split checks, deterministic schedules, checkpoint identity verification, normal-speech controls, and separate hypothesis diagnostics. See the [frozen configuration](https://github.com/sethanimesh/echora/blob/main/research/training/configs/qwen_command_v3.json) and [training implementation](https://github.com/sethanimesh/echora/blob/main/research/training/qwen_command_v3.py).

## Training data and evaluation design

| Command partition | Speakers | Commands | Audio hours |
| --- | --- | ---: | ---: |
| Training | F01, F04, M01, M02, M03, M05 | 6,000 | 14.6240 |
| Development | F03 | 480 | 0.9643 |
| Held-out test | M04 | 480 | 1.2331 |

Controlled two- and three-word commands are composed from genuine TORGO isolated-word recordings, with gaps up to 3.2 seconds and time stretching up to 2x. TORGO utterances provide dysarthric controls, and Common Voice provides normal-speech controls.

Commands are controlled augmentations rather than naturally spoken requests. The folds contain eight dysarthric speakers. Command partitions are disjoint by speaker and composed phrase. Original TORGO prompt texts may overlap across speakers, as disclosed in the configuration; the TORGO evaluation is a speaker holdout rather than an unseen-phrase or population study.

## Artifacts and inference

The release contains `adapter.safetensors` and `adapter_config.json`. The weight file includes tuned acoustic and projector parameters alongside decoder LoRA state. Use Echora's loader for this composite adapter format.

| Artifact identity | Value |
| --- | --- |
| Foundation repository | `Qwen/Qwen3-ASR-1.7B-hf` |
| Foundation revision | `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` |
| Adapter release tag | `v3` |
| Foundation SHA-256 | `2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1` |
| Adapter SHA-256 | `7cd203cc0cbf479e6afa198cc3895fedc71f1907b870fcb7f1dece0a1e6b2021` |

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

The literal instruction is part of the measured decoding contract:

> Transcribe only the words actually spoken. Keep incomplete or unusual word sequences literal. Do not add missing words.

The communication application requests five hypotheses and preserves their sequence scores. Relative beam-search weights are not calibrated confidence. The separate application verification scorer and pretrained foundation are not included in this adapter repository.

## Intended use and limitations

The intended use is research on assistive communication with visible recognition alternatives and speaker control. Clinical benefit, population-level recognition performance, and suitability for an individual have not been established. Recognition can omit or substitute important words, and performance varies by speaker and recording conditions.

The application separates recognition, acoustic verification, wording generation, and revision-bound speech authorization. Its current learned automatic-selection audit failed, so that route requests a choice; this does not alter the adapter's ASR measurements. See the [application architecture](https://github.com/sethanimesh/echora/blob/main/docs/architecture.md) for the interaction policy.

<details>
<summary>Detailed evaluation, adaptation trade-offs, and historical baselines</summary>

### Normal-speech development control

The same pretrained-foundation comparison measured **3.44% WER** before adaptation and **4.47% WER** after adaptation on 244 normal-speech development recordings. Targeted dysarthric and command improvements therefore accompany increased normal-speech WER under this development protocol.

### Historical adapter comparison

The previous internal adapter served as the baseline for the protected test and separate beam diagnostic. It is included here for traceability rather than as an additional release recommendation.

| Protected-test measure | Previous adapter | Released adapter |
| --- | ---: | ---: |
| TORGO WER, M04 | 56.29% | 55.02% |
| Composed-command WER, M04 | 72.58% | 51.58% |
| Normal-speech WER | 5.23% | 5.23% |

The composed-command WER difference corresponds to a **28.9% relative reduction**. In the separate five-beam diagnostic, exact-reference coverage increased from **16.04% to 36.46%**.

### Speaker-specific smoke test

Three personal utterances, excluded from training and selection, yielded **22.22% WER** for the previous adapter and **44.44% WER** for the released adapter. This is a documented speaker-specific regression, not a population estimate.

The incomplete phrase `I water` is absent from both adapters' top-five hypotheses. The released adapter produces `thigh button`, `thigh gotten`, `high button`, `nine button`, and `thigh bottom`. Multiple hypotheses cannot recover a reference that is absent from every candidate.

The full [experiment result](https://github.com/sethanimesh/echora/blob/main/models/echora-qwen3-asr-command-v3/evaluation/result.json) and [deployment decision](https://github.com/sethanimesh/echora/blob/main/models/echora-qwen3-asr-command-v3/evaluation/next_decision.json) preserve all measurements. Later acoustic-verification results use a different decoding protocol and do not demonstrate a new improvement in these adapter weights.

</details>

Raw recordings and private profile stores are excluded from the GitHub source repository. Further [failure analysis](https://github.com/sethanimesh/echora/blob/main/docs/failure-analysis.md) and [evaluation limitations](https://github.com/sethanimesh/echora/blob/main/docs/limitations.md) describe the broader application boundaries.
