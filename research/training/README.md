# Universal dysarthric ASR training

This directory is the acoustic-recognition workstream. It produces literal
transcripts and raw decoder alternatives. Personal context, message cleanup,
and the Echora candidate ranker are downstream and must never change labels,
splits, decoding scores, or WER/CER reported here.

## Prepare TORGO

From the repository root:

    .venv/bin/python research/training/prepare_torgo.py

The command selects only the head microphone, pairs audio with genuine lexical
prompts, and writes ignored artifacts under `data/derived/torgo/`. It excludes
image-prompted spontaneous recordings because TORGO does not provide the words
actually spoken in those prompt files. It also excludes mouth-position and
sustained-vowel instructions.

Every dysarthric speaker gets an outer test fold. The next dysarthric speaker is
the development speaker; no speaker occurs in two partitions. Each fold has:

- `train/dev/test.jsonl`: the normal unseen-speaker experiment;
- `train_no_test_text/dev_no_test_text.jsonl`: stricter training/development
  manifests with every test-speaker prompt removed, preventing the repeated
  TORGO phrase list from flattering results.

Do not select a single convenient fold. Report all outer dysarthric-speaker
folds and macro-average the per-speaker scores.

## Foundation selected

The fixed 400-utterance foundation gate selected official Qwen3-ASR 1.7B:
41.90% speaker-macro WER versus 45.83% for Parakeet-TDT 1.1B, with fewer
deletions and no empty outputs. The first paid adaptation therefore targets
Qwen only. Parakeet and CTC remain frozen baselines, not parallel training jobs.

Qwen's official repository provides a JSONL supervised fine-tuning script with
checkpoint resume. Echora will first wrap that supported path in a bounded
forward/backward and checkpoint probe. The first partial-tuning configuration
must update acoustic encoder layers and the multimodal projector; tuning only
the language decoder would not directly adapt dysarthric acoustics.

Gate 2 subsequently passed: 2.63% of parameters trained, tiny-probe loss fell
37.70%, peak L40S memory was 12.86 GiB, and a fresh reload resumed successfully.
That proves the mechanics only, not WER improvement.

The complete next pilot and copy-paste L40S instructions are in
`research/training/PILOT-README.md`. It uses the full speaker-disjoint TORGO training
fold, required Common Voice retention data, development-WER checkpoint
selection, a protected outer test, adapter export and fresh-reload validation.

Fine-tune the foundation model; do not train these billion-parameter encoders
from scratch on TORGO. Balance by speaker rather than raw utterance count, since
TORGO ranges from roughly one hundred to one thousand head-mic clips per
speaker. Mix in ordinary English retention audio before judging a universal
model.

`augment.py` implements conservative training-only speed, pause, noise, and gain
transforms. Pause insertion targets an existing low-energy internal boundary.
Never augment development/test audio and never label generated audio as a new
speaker. Room impulse responses and real noise clips should replace synthetic
white noise once those assets are available.

Before a longer run, add ordinary-English retention data and freeze the exact
train/dev manifests. Use an unseen dysarthric development speaker, never the
outer test speaker, for checkpoint selection.

## Archived CTC implementation

Audit one outer fold locally without loading the 1.1B model:

    .venv/bin/python research/training/finetune_parakeet_ctc.py \
      --train-manifest data/derived/torgo/folds/M04/train.jsonl \
      --dev-manifest data/derived/torgo/folds/M04/dev.jsonl \
      --output data/derived/runs/ctc-M04 \
      --dry-run

`finetune_parakeet_ctc.py` remains reproducible comparator code, but the
foundation gate does not authorize spending the first pilot budget on it.

This is one development run, not the final number. Repeat all eight outer test
speakers after hyperparameters are frozen. For the stricter experiment, switch
to `train_no_test_text.jsonl` and `dev_no_test_text.jsonl`; do not compare its
WER directly with the easier repeated-prompt split.
