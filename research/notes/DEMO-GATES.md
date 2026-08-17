# Demo-first experiment gates

The objective is a convincing literal-ASR demo without spending the $20 budget
on an unvalidated training recipe. Passing a gate authorizes preparation of the
next gate; paid compute still requires explicit user approval.

## Budget

- Hard total cloud ceiling: **$20**, including GPU and cloud storage.
- Local preparation and locally cached inference happen before paid compute.
- Cloud gate 1 is a frozen, foundation-only inference comparison capped at
  **$1**. The selected model then receives a separate forward/backward,
  checkpoint and resume probe before training.
- The first training run is capped at **$8 cumulative spend**. If development
  loss is invalid or WER has not improved, stop and diagnose before spending
  more.
- At least **$4 remains reserved** for final evaluation and demo inference.
- No automatic GPU/model/dataset substitution is allowed after a failed gate.

## Data integrity

- Train, development and test speakers are disjoint.
- The strict condition also removes every outer-test prompt from train/dev.
- Only train audio receives augmentation.
- TORGO head microphone is used once per utterance; duplicate array-microphone
  recordings and prompt files that are not literal transcripts stay excluded.
- Normal-English retention data is measured separately and never merged into
  the dysarthric headline score.

## Candidate selection

- Measure literal zero-shot WER/CER and per-speaker macro WER before training.
- Compare only official foundation checkpoints: Parakeet-TDT 1.1B and
  Qwen3-ASR 1.7B. Community dysarthria adapters and fine-tunes are excluded.
- The first screen contains 50 deterministic, duration-stratified utterances
  from each of the eight dysarthric TORGO speakers. Neither model advances on
  the three personal clips alone.
- A screening winner needs at least 5% relative speaker-macro WER improvement
  without increasing either deletion rate or empty-output rate by more than two
  absolute points. Otherwise both models receive full-TORGO evaluation.
- CTC remains a comparator/auxiliary objective.
- Partial tuning must reach acoustic encoder layers; decoder/head-only tuning is
  not accepted as the main experiment.

The official 1.1B `.nemo` archive is staged at
`/Volumes/Extreme Pro/echora/checkpoints/base-models/nvidia-parakeet-tdt-1.1b/`
and matches its published SHA-256
`9c563d52bdffeacbac0c5b894fdea9be82fea3a6bd8bb8018ff57888e2b5d988`.
Its embedded configuration has 42 FastConformer encoder layers plus the TDT
decoder and joint network.

The completed stock Parakeet-TDT 0.6B v3 TORGO baseline is 53.4% micro WER,
55.9% speaker-macro WER and 15.3% empty outputs. Speaker WER ranges from 12.0%
to 86.1%. The foundation gate therefore includes every speaker and does not
hide the hardest speaker. Training and checkpoint selection still use
speaker-disjoint folds fixed before adaptation.

## Paid gate 1 — complete

- Official Qwen3-ASR 1.7B: 41.90% speaker-macro WER, 2.30% deletions, and no
  empty outputs on the fixed 400-utterance gate.
- Official Parakeet-TDT 1.1B: 45.83% speaker-macro WER, 5.84% deletions, and
  1.25% empty outputs on the identical IDs.
- Qwen's relative speaker-macro improvement is 8.57%; it passes the frozen 5%
  selection margin and both guardrails. Qwen is the provisional foundation.
- No backward pass was performed. The candidate ranker was not involved.
- The two models ran on different devices, so latency was not compared.
- The RunPod volume retained `/workspace`, while the restarted container lost
  `ffmpeg`; the dependency was reinstalled before the successful resumable run.
- Two terminals briefly launched the same resumable evaluation. All 400 paired
  duplicates had identical references and transcripts. The raw concurrent copy
  was preserved and the scored TSV was validated at 400 unique IDs.

## Paid gate 2 — complete

- Qwen3-ASR 1.7B audio layers 20–23 plus the multimodal projector trained on an
  L40S; the decoder and remaining foundation parameters stayed frozen.
- 53,533,696 of 2,038,052,480 parameters were trainable (2.63%).
- Tiny-probe mean loss fell from 2.9023 to 1.8082, a 37.70% relative reduction.
- A fresh foundation reload restored the partial checkpoint and completed step
  19 at loss 1.6440.
- Peak allocated GPU memory was 12.86 GiB. Training completed without non-finite
  loss or OOM.
- This passed the trainability gate but did not measure adapted WER.

No alternate GPU, model, dataset, or training scope is selected automatically
after a failed gate.

The full WER-measuring pilot implementation and copy-paste handoff are ready in
`research/training/PILOT-README.md`. It requires independently prepared Common Voice
retention data and separate approval before another paid run.

## Pilot success

For the demo-oriented first pilot, the complete speaker-disjoint TORGO fold is
used to maximize dysarthric acoustic coverage. Because TORGO repeats prompts
between speakers, its M04 result is explicitly labelled an unseen-speaker,
repeated-prompt condition. The stricter prompt-excluded experiment remains
required before making research claims about lexical generalization.

On the untouched development speaker, the adapted model must:

- reduce macro WER by at least 15% relative to its own frozen baseline;
- not degrade the normal-English retention WER by more than 2 absolute points;
- produce finite loss, no transcript leakage and a resumable checkpoint;
- avoid increased empty-output or silence-hallucination rates;
- retain the literal ASR output before any Echora semantic candidate ranker.

The three personal clips and their synthetic pause variants are external smoke
tests. They are displayed in the demo but are too small to select checkpoints.

## SSD contract (confirmed)

Locations:

- `/Volumes/Extreme Pro/echora/configs/<run-id>/` — frozen hyperparameters only.
- `/Volumes/Extreme Pro/echora/checkpoints/<run-id>/` — model/adapter weights and
  the state required to resume training.

Datasets, augmented audio, logs, predictions and benchmark reports remain in
the local project. Existing SSD contents are left untouched.
