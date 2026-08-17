# Qwen TORGO pilot — complete runbook

This is the next experiment after the successful Gate 2 mechanics probe. It is
an end-to-end literal speech-to-text run, not another six-recording test.

The fixed run does all of the following without manual model decisions:

1. validates the official Qwen3-ASR 1.7B foundation by revision, byte count and
   SHA-256;
2. checks that TORGO speakers are disjoint across train, development and test;
3. uses every eligible dysarthric recording from all six training speakers in
   every epoch;
4. rotates through the TORGO control pool and mixes speaker-balanced Common
   Voice normal speech;
5. applies conservative timing, pause, noise and gain augmentation only to
   dysarthric training audio;
6. measures frozen-Qwen F03 and normal-English baselines before training;
7. partially trains audio layers 20–23 plus the multimodal projector, leaving
   the language decoder and remaining foundation parameters frozen;
8. saves a resumable checkpoint every 50 optimizer steps;
9. evaluates literal F03 WER and normal-English retention after each epoch;
10. chooses the best checkpoint only when WER, deletion, empty-output and
    normal-speech guardrails pass;
11. keeps M04 sealed unless the best F03 checkpoint improves WER by at least
    15% relative;
12. evaluates M04 and independent normal test speech once, exports a standalone
    adapter, reloads it over a fresh foundation and verifies identical personal
    smoke-test transcripts.

A final `passed` additionally requires at least 5% relative M04 WER improvement
without increasing its deletion or empty-output rate by more than two absolute
points. M04 never changes which checkpoint is selected; it determines only the
honest final verdict.

There is no candidate ranker, personal-context correction, CTC auxiliary loss,
language-model rescoring or automatic fallback in this pipeline.

Frozen configuration SHA-256:
`698db922c8a86ab05deb39a9776ad789378008e622bf671a1fe75752de715a04`.
The identical config is stored at
`/Volumes/Extreme Pro/echora/configs/qwen3-asr-1.7b-torgo-pilot-v1/`.

## What “whole TORGO dataset” means here

The full speaker-disjoint M04 fold is used, not the reduced six-recording probe
and not the 697-utterance prompt-excluded subset. After the fixed 30-second
quality limit, it contains approximately:

- 2,043 dysarthric recordings from F01, F04, M01, M02, M03 and M05;
- 5,058 TORGO control recordings;
- 557 F03 development recordings;
- 281 M04 outer-test recordings.

Every dysarthric training recording appears exactly once per epoch. Exactly 633
control recordings rotate through each epoch, so an eight-epoch run covers the
entire control pool without allowing normal speech to overwhelm the target
speech. Three hundred speaker-balanced Common Voice clips are added per epoch.

TORGO repeats its prompt list across speakers. Using the full fold maximizes the
available dysarthric acoustics for the demo, but the M04 number is therefore an
**unseen-speaker, repeated-prompt** result—not proof of universal stroke ASR.
The scripts record this limitation in the frozen config and result metadata.

## 1. Run the local preflight tests

Start in the project root with the SSD attached:

    cd "/Users/animesh/Animesh/Project 2.0/echora"
    test -d "/Volumes/Extreme Pro/echora"
    shasum -a 256 \
      research/training/configs/qwen_torgo_pilot_v1.json \
      "/Volumes/Extreme Pro/echora/configs/qwen3-asr-1.7b-torgo-pilot-v1/qwen_torgo_pilot_v1.json"

Both config hashes must be:

    698db922c8a86ab05deb39a9776ad789378008e622bf671a1fe75752de715a04

Run every local test and syntax check:

    .venv/bin/python -m unittest discover \
      -s research/training/tests -p 'test_*.py'
    .venv/bin/python -m unittest discover \
      -s research/benchmarks/tests -p 'test_*.py'
    python3 -m py_compile \
      research/training/prepare_common_voice.py \
      research/training/stage_qwen_pilot.py \
      research/training/qwen_torgo_pilot.py \
      research/training/run_qwen_adapter.py
    bash -n \
      research/training/cloud/prepare_qwen_pilot_bundle.sh \
      research/training/cloud/setup_qwen_pilot_l40s.sh \
      research/training/cloud/run_qwen_pilot.sh

Expected unit-test totals are 20 training tests and 9 benchmark tests, both
ending in `OK`. Do not create a paid Pod if any command fails.

Confirm the three personal references are still correct:

    for file in research/benchmarks/clips/*.txt; do echo "$(basename "$file"): $(cat "$file")"; done

They must read `I want water`, `I water`, and `I want water please`.

## 2. Prepare the required normal-English data locally

The pilot intentionally refuses to start without independent normal speech.
Follow `data/raw/common_voice/README.md` and extract one or more English Common
Voice subsets anywhere below:

    data/raw/common_voice/

The builder needs the standard `validated.tsv` (or train/dev/test TSV files)
and their `clips/` directory. From the project root run:

    cd "/Users/animesh/Animesh/Project 2.0/echora"
    .venv/bin/python research/training/prepare_common_voice.py \
      --source data/raw/common_voice \
      --output data/derived/common_voice/pilot-v1 \
      --train-hours 3 \
      --dev-minutes 20 \
      --test-minutes 20

The output must contain:

    data/derived/common_voice/pilot-v1/train.jsonl
    data/derived/common_voice/pilot-v1/dev.jsonl
    data/derived/common_voice/pilot-v1/test.jsonl
    data/derived/common_voice/pilot-v1/summary.json

Client IDs are converted to one-way hashes. Train/dev/test remain
speaker-disjoint. If the available subset cannot meet the requested hours, the
builder fails instead of silently reducing the retention test.

Inspect the generated split summary before continuing:

    python3 -m json.tool data/derived/common_voice/pilot-v1/summary.json
    wc -l data/derived/common_voice/pilot-v1/*.jsonl

The summary must report at least 3 hours of train speech and 20 minutes each for
development and test. The speaker counts must be nonzero in all three splits.

## 3. Build and inspect the complete upload archive

From the project root:

    chmod +x research/training/cloud/prepare_qwen_pilot_bundle.sh
    research/training/cloud/prepare_qwen_pilot_bundle.sh

The command audits and copies all selected audio, rewrites paths for RunPod and
creates:

    /private/tmp/echora-qwen-pilot-v1.tar.gz

Expect a large archive because it contains hours of audio. The official 4 GB
Qwen foundation is not included; setup downloads and verifies it on the Pod.

Verify the archive before paying for a GPU:

    ls -lh /private/tmp/echora-qwen-pilot-v1.tar.gz
    tar -tzf /private/tmp/echora-qwen-pilot-v1.tar.gz \
      | grep -E 'stage_report.json|qwen_torgo_pilot.py|run_qwen_pilot.sh'

All three names must appear. The bundle builder has already audited the real
audio files, duration limit, speakers, IDs and manifest relationships.

## 4. Create one RunPod L40S

Create one L40S Pod with SSH enabled and a persistent `/workspace` volume. Do
not select a different GPU. There is no automatic shutdown; you stop the Pod
after the outputs are safely copied.

Use this GPU checklist:

- GPU: exactly one NVIDIA L40S (48 GB class; the script verifies the name).
- Persistent volume mounted at `/workspace`.
- At least 50 GB free in `/workspace` for the archive, isolated Python/CUDA
  packages, the 4 GB foundation, temporary files and checkpoints.
- SSH public-key access enabled.
- An Ubuntu/PyTorch-style template with Python 3 and `apt-get`; the system
  PyTorch version does not matter because setup creates an isolated venv.

After the Pod starts, do not begin setup until RunPod shows it as running and
provides the direct SSH command.

Use the current RunPod endpoint and port below in all commands:
`root@103.196.86.40` on port `47175`.

## 5. Upload the archive and verify the GPU

On the Mac:

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 47175 \
      /private/tmp/echora-qwen-pilot-v1.tar.gz \
      root@103.196.86.40:/workspace/

Then connect:

    ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 \
      root@103.196.86.40 -p 47175

Inside the Pod, verify the exact hardware and storage before installing:

    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    df -h /workspace
    ls -lh /workspace/echora-qwen-pilot-v1.tar.gz

The first command must show `NVIDIA L40S`. If it shows another GPU, stop here;
do not edit the configuration to accept it.

## 6. Extract, install and test the GPU environment

Inside the Pod:

    mkdir -p /workspace/echora
    tar --no-same-owner -xzf /workspace/echora-qwen-pilot-v1.tar.gz \
      -C /workspace/echora
    chmod +x /workspace/echora/research/training/cloud/setup_qwen_pilot_l40s.sh
    chmod +x /workspace/echora/research/training/cloud/run_qwen_pilot.sh
    /workspace/echora/research/training/cloud/setup_qwen_pilot_l40s.sh \
      2>&1 | tee /workspace/echora/qwen_pilot_setup.log

Success ends with:

    QWEN_PILOT_SETUP_DONE: isolated environment and exact Qwen foundation verified on NVIDIA L40S

The environment is isolated from the RunPod image’s system PyTorch and cuDNN.
The setup success marker is printed only after importing the pinned packages,
finding CUDA and completing a real matrix multiplication on the L40S.

Run this explicit post-setup test as well:

    /workspace/venv-qwen-pilot-isolated/bin/python -c \
      "import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GiB')"
    sha256sum /workspace/models/Qwen3-ASR-1.7B-hf/model.safetensors

Expected versions are PyTorch `2.6.0+cu124`, Transformers `5.15.0`, and GPU
`NVIDIA L40S`. Transformers 5.15.0 requires Safetensors 0.8.0; the setup pins
and verifies that exact version. The model SHA-256 must be:

    2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1

If setup fails, do not change package versions. Paste:

    tail -n 150 /workspace/echora/qwen_pilot_setup.log

## 7. Start this batch from zero

Inside the Pod:

    /workspace/echora/research/training/cloud/run_qwen_pilot.sh fresh \
      2>&1 | tee -a /workspace/echora/qwen_pilot.log

`fresh` always loads the verified official Qwen foundation. It does not load
the Gate 2 probe or any previous RunPod checkpoint. Gate 2 uses a different run
directory, but the launcher also refuses to start if it finds existing state
for this pilot run ID. It never deletes or replaces that state automatically.

The first lines must include:

    QWEN_PILOT_FRESH: loading the verified official foundation; no checkpoint will be restored

The terminal is the completion hook; no polling is needed. It ends with one of:

    QWEN_PILOT_DONE: result, adapter and resumable checkpoints are in /workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1

or:

    QWEN_PILOT_FAILED: paste result.json if present, otherwise the final 150 log lines

Failure can be a valid scientific result—for example, less than 15% relative
F03 WER improvement. The script will not unseal M04 or change the recipe to
force a pass.

If SSH disconnects, reconnect with the same SSH command and first check whether
training is still alive:

    pgrep -af qwen_torgo_pilot.py

If a process is shown, do not launch another run; inspect the existing log. If
no process is shown and this new batch produced `checkpoint-latest.pt`, resume
it explicitly:

    /workspace/echora/research/training/cloud/run_qwen_pilot.sh resume \
      2>&1 | tee -a /workspace/echora/qwen_pilot.log

`resume` verifies the foundation, config and all seven manifest hashes before
restoring anything. Do not use `resume` for the old Gate 2 run. Do not delete
the output directory or change arguments. To see the last recorded state:

    tail -n 30 /workspace/echora/qwen_pilot.log
    ls -l /workspace/echora/status/qwen3-asr-1.7b-torgo-pilot-v1

## 8. Inspect the automatic test result

The run itself performs all important comparisons. Inspect the final report:

    python3 -m json.tool \
      /workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/result.json

For a successful run, verify these fields rather than relying only on the word
`passed`:

- `relative_dev_wer_improvement` is at least `0.15`;
- `relative_torgo_test_wer_improvement` is at least `0.05`;
- `normal_test_wer_degradation_absolute` is at most `0.02`;
- `outer_test_guard_failures` is empty;
- `adapter_reload_transcripts_identical` is `true`;
- `personal_adapted.utterances` is `3`.

The literal per-clip outputs are under:

    /workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/predictions/

Compare the three personal transcripts directly:

    column -t -s $'\t' \
      /workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/predictions/baseline_personal.tsv
    column -t -s $'\t' \
      /workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/predictions/adapted_personal.tsv

If `column` is unavailable, use `sed -n '1,5p'` on each file. Personal clips
are a smoke test, not the checkpoint-selection metric.

## 9. Copy and verify artifacts before stopping the Pod

On the Mac, create the destinations:

    cd "/Users/animesh/Animesh/Project 2.0/echora"
    mkdir -p research/training/results/qwen3-asr-1.7b-torgo-pilot-v1
    mkdir -p "/Volumes/Extreme Pro/echora/configs/qwen3-asr-1.7b-torgo-pilot-v1"
    mkdir -p "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1"

Copy the small reports locally:

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 47175 \
      root@103.196.86.40:/workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/result.json \
      root@103.196.86.40:/workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/run_manifest.json \
      root@103.196.86.40:/workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/training_history.json \
      research/training/results/qwen3-asr-1.7b-torgo-pilot-v1/

Copy the full literal prediction files and cloud log as well:

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 47175 -r \
      root@103.196.86.40:/workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/predictions \
      research/training/results/qwen3-asr-1.7b-torgo-pilot-v1/
    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 47175 \
      root@103.196.86.40:/workspace/echora/qwen_pilot.log \
      research/training/results/qwen3-asr-1.7b-torgo-pilot-v1/

Copy the deployable adapter and frozen config to the SSD:

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 47175 \
      root@103.196.86.40:/workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors \
      root@103.196.86.40:/workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/adapter_config.json \
      "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/"

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 47175 \
      root@103.196.86.40:/workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/qwen_torgo_pilot_v1.json \
      "/Volumes/Extreme Pro/echora/configs/qwen3-asr-1.7b-torgo-pilot-v1/"

For continued training, also copy both resumable files to the SSD:

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 47175 \
      root@103.196.86.40:/workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/checkpoint-best.pt \
      root@103.196.86.40:/workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/checkpoint-latest.pt \
      "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/"

Only stop the Pod after the local sizes and the SHA-256 values recorded in
`result.json` match. The scripts never stop or terminate it automatically.

Verify the copied adapter on the Mac:

    shasum -a 256 \
      "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors"

Its value must equal `adapter.adapter_sha256` in `result.json`. Verify both
resumable checkpoint hashes the same way when those files were copied.

## 10. Test the adapted model on the three personal clips

On any CUDA machine with the same environment and verified Qwen foundation:

    /workspace/venv-qwen-pilot-isolated/bin/python \
      /workspace/echora/research/training/run_qwen_adapter.py \
      --model /workspace/models/Qwen3-ASR-1.7B-hf \
      --adapter /workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors \
      --audio /workspace/echora/pilot_data/audio/personal \
      --output /workspace/echora/personal-adapted.tsv

This emits literal top-1 transcripts and optional WER when matching `.txt`
sidecars exist. It does not rewrite the output or invoke an Echora ranker.

The output should show each filename, reference and transcript, followed by WER
and CER across the three clips. Inspect it with:

    column -t -s $'\t' /workspace/echora/personal-adapted.tsv

## 11. Test deliberate long-pause variants

On the Mac, generate fixed 0.5, 1.0 and 2.0-second pause-stress copies of the
three referenced clips:

    cd "/Users/animesh/Animesh/Project 2.0/echora"
    .venv/bin/python research/benchmarks/make_pause_stress.py \
      --clips research/benchmarks/clips \
      --output data/derived/pause_stress \
      --seconds 0.5 1.0 2.0
    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 47175 -r \
      data/derived/pause_stress root@103.196.86.40:/workspace/echora/

Inside the Pod, transcribe each stress level:

    for delay in 0500ms 1000ms 2000ms; do
      /workspace/venv-qwen-pilot-isolated/bin/python \
        /workspace/echora/research/training/run_qwen_adapter.py \
        --model /workspace/models/Qwen3-ASR-1.7B-hf \
        --adapter /workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors \
        --audio "/workspace/echora/pause_stress/${delay}" \
        --output "/workspace/echora/pause-stress-${delay}.tsv"
    done

Each run prints WER/CER because the generated WAV files retain their `.txt`
references. Copy the three TSVs with the other reports. These are external
stress tests and never influence checkpoint selection.

## 12. Test a new recording

Place a WAV plus an optional same-named transcript sidecar in a local folder,
for example `new-test/example.wav` and `new-test/example.txt`. Upload it:

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 47175 -r \
      new-test root@103.196.86.40:/workspace/echora/

Then run inside the Pod:

    /workspace/venv-qwen-pilot-isolated/bin/python \
      /workspace/echora/research/training/run_qwen_adapter.py \
      --model /workspace/models/Qwen3-ASR-1.7B-hf \
      --adapter /workspace/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors \
      --audio /workspace/echora/new-test \
      --output /workspace/echora/new-test-results.tsv

Keep pauses and stretched speech intact; do not trim silence to make the test
easier. If `.txt` exists, the utility reports literal WER/CER. Without it, it
still writes the raw transcript.

## 13. Stop the GPU

After the adapter, config, result and any desired resumable checkpoints are
verified locally, stop or terminate the Pod from RunPod. The project scripts do
not perform this action for you.

## Expected duration

Exact time depends on the Common Voice clips and RunPod storage speed. A
reasonable L40S planning range is:

- environment/model setup: 10–20 minutes on a warm network;
- frozen baselines: 10–30 minutes;
- three to eight training epochs plus development decoding: roughly 1.5–4
  hours;
- protected final evaluation and reload verification: 15–40 minutes.

Treat these as planning ranges, not promises. Early stopping can finish after
three epochs; the hard configuration cannot exceed eight. Cost is the Pod’s
displayed hourly rate multiplied by actual wall time. Do not start unless the
rate and remaining budget can cover the upper range.
