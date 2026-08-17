# Qwen command v3: one complete literal-ASR experiment

This is the next full experiment for the data we can actually obtain. It does
not wait for another corpus and it does not run a series of paid micro-tests.

The run starts from the exact official Qwen3-ASR 1.7B foundation. The verified
v1 adapter is used only as the baseline and fallback; v3 never resumes v1 or
v2 training state.

## What v3 changes

- Uses every eligible original TORGO dysarthric training utterance each epoch.
- Adds a pool of 6,000 same-speaker, literal two/three-word compositions made
  from real TORGO dysarthric isolated-word recordings.
- Balances short, medium, long and very-long gaps up to 3.2 seconds.
- Balances no, mild, strong and very-strong word stretching up to 2x.
- Keeps F03 as development and M04 as the protected outer test speaker.
- Makes command phrases disjoint across train, development and test.
- Tunes only Qwen audio layers 20–23, the multimodal projector, and rank-8 LoRA
  updates on q/v attention projections in decoder layers 20–27.
- Uses a fixed literal instruction: preserve incomplete/unusual word sequences
  and do not add missing words.
- Reports literal top 1 and five raw beam alternatives.

There is no candidate ranker, grammar correction, intended-message recovery,
or personal context. The three personal clips never enter training or model
selection.

The composed commands are controlled augmentation, not naturally spoken new
clinical data. They can test and train literal sequencing, pauses and stretch,
but they do not increase the number of dysarthric speakers.

## Real staged data audit

| Split | Speaker(s) | Commands | Hours | Two/three words |
| --- | --- | ---: | ---: | ---: |
| Train | F01, F04, M01, M02, M03, M05 | 6,000 | 14.6240 | 3,000 / 3,000 |
| Development | F03 | 480 | 0.9643 | 240 / 240 |
| Protected test | M04 | 480 | 1.2331 | 240 / 240 |

Each split has equal coverage of four gap classes and four stretch classes.
There is no speaker or composed-phrase overlap between splits.

## Expected GPU time and cost

Use one NVIDIA L40S with 48 GB VRAM. Allow approximately 4–7 hours after setup
for baselines, up to eight training epochs, protected tests, and top-five
diagnostics. Early stopping may finish sooner.

RunPod currently lists L40S Pods around $0.99/hour, so the expected compute is
roughly $4–$7, with substantial room inside the earlier $20 budget. Upload,
model download and idle time still cost money while the Pod is running.

No shutdown or Pod deletion command is installed. Stop the Pod manually only
after copying and verifying the result directory on the SSD.

## Step 1 — Use or rebuild the Mac archive

The prepared archive is:

```text
/private/tmp/echora-qwen-command-v3.tar.gz
```

Check that it exists and record its current checksum:

```bash
ls -lh /private/tmp/echora-qwen-command-v3.tar.gz
shasum -a 256 /private/tmp/echora-qwen-command-v3.tar.gz
```

If it is missing, attach the SSD and rebuild it from any Mac directory:

```bash
cd "/Users/animesh/Animesh/Project 2.0/echora"

./research/training/cloud/prepare_qwen_command_v3_bundle.sh \
  /private/tmp/echora-qwen-command-v3.tar.gz \
  "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors"
```

The builder verifies the v1 SHA-256, stages all data, generates all command
audio, audits the splits, and prints the archive checksum. Building takes a few
minutes and produces an archive of about 2.2 GB.

## Step 2 — Upload to a fresh L40S Pod

Set the host and port shown by RunPod in the same Mac terminal:

```bash
COMMAND_V3_HOST=202.181.159.233
COMMAND_V3_PORT=19317
```

Upload from the Mac:

```bash
scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 \
  -P 19317 \
  /private/tmp/echora-qwen-command-v3.tar.gz \
  root@202.181.159.233:/workspace/
```

Connect:

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 \
  -p 19317 \
  root@202.181.159.233
```

Direct command:

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -p 19317 root@202.181.159.233
```

## Step 3 — Extract and set up the Pod

Run these commands on the Pod:

```bash
mkdir -p /workspace/echora

tar --no-same-owner \
  -xzf /workspace/echora-qwen-command-v3.tar.gz \
  -C /workspace/echora

chmod +x \
  /workspace/echora/research/training/cloud/setup_qwen_command_v3_l40s.sh \
  /workspace/echora/research/training/cloud/run_qwen_command_v3.sh

/workspace/echora/research/training/cloud/setup_qwen_command_v3_l40s.sh
```

Setup must end with:

```text
COMMAND_V3_SETUP_DONE: exact environment and foundation verified on NVIDIA L40S
```

The setup refuses a non-L40S GPU, a non-isolated environment, or foundation
weights with a different size or SHA-256.

## Step 4 — Start the complete run

On the Pod:

```bash
/workspace/echora/research/training/cloud/run_qwen_command_v3.sh fresh
```

Keep the terminal open. Live output is also written to:

```text
/workspace/echora/checkpoints/qwen3-asr-1.7b-command-v3/command_v3.log
```

The runner checkpoints every 50 optimizer steps and after every epoch. The
completion hook is:

```text
COMMAND_V3_DONE: copy ... to the SSD and paste SHARE_THIS_WITH_CODEX.txt
```

## If SSH disconnects or the run is interrupted

Reconnect and run:

```bash
/workspace/echora/research/training/cloud/run_qwen_command_v3.sh resume
```

Resume verifies the run ID, foundation, v1 baseline, config and every manifest.
Do not change the configuration between `fresh` and `resume`.

If it prints `COMMAND_V3_FAILED`, paste this output here without modifying the
recipe:

```bash
tail -n 180 \
  /workspace/echora/checkpoints/qwen3-asr-1.7b-command-v3/command_v3.log
```

## Step 5 — Inspect the result on the Pod

```bash
cat /workspace/echora/status/qwen3-asr-1.7b-command-v3/DONE

cat \
  /workspace/echora/checkpoints/qwen3-asr-1.7b-command-v3/SHARE_THIS_WITH_CODEX.txt
```

Paste the complete `SHARE_THIS_WITH_CODEX.txt` into this task. It contains:

- v1 versus v3 M04 TORGO WER;
- v1 versus v3 M04 command top-one WER;
- v1 versus v3 command top-five exact coverage;
- normal-speech retention;
- all v1/v3 alternatives for “I water”;
- the automatic `deploy_v3_literal_multi_hypothesis` or `keep_v1` decision.

## Step 6 — Test the selected v3 adapter on the Pod

This command returns literal top-five alternatives for the personal clips:

```bash
/workspace/venv-qwen-command-v3/bin/python \
  /workspace/echora/research/training/run_qwen_command_adapter.py \
  --config /workspace/echora/research/training/configs/qwen_command_v3.json \
  --model /workspace/models/Qwen3-ASR-1.7B-hf \
  --adapter /workspace/echora/checkpoints/qwen3-asr-1.7b-command-v3/adapter.safetensors \
  --audio /workspace/echora/pilot_data/audio/personal \
  --output /workspace/echora/command-v3-personal.tsv \
  --beams 5
```

The probabilities in the TSV are relative beam-search scores, not calibrated
confidence values. Alternatives are not semantically re-ranked.

## Step 7 — Copy everything to the SSD

Run on the Mac in the terminal where the Pod connection is configured:

```bash
mkdir -p "/Volumes/Extreme Pro/echora/checkpoints"

scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 \
  -P 19317 -r \
  root@202.181.159.233:/workspace/echora/checkpoints/qwen3-asr-1.7b-command-v3 \
  "/Volumes/Extreme Pro/echora/checkpoints/"
```

Verify the important files:

```bash
find "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-command-v3" \
  -maxdepth 2 -type f -print | sort

shasum -a 256 \
  "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-command-v3/adapter.safetensors" \
  "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-command-v3/checkpoint-latest.pt" \
  "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-command-v3/checkpoint-best-candidate.pt"
```

If `checkpoint-best-deployable.pt` exists, checksum that too. Stop the Pod only
after the SSD copy is complete.

## Checkpoint and deployment rules

Every epoch is scored on F03 original speech, F03 composed commands, and normal
development speech. A checkpoint is deployable only when it:

- improves F03 command WER by at least 5% relative to v1;
- stays within 0.02 absolute WER of v1 on original F03 TORGO;
- stays within 0.02 absolute WER of v1 on normal speech;
- does not materially increase deletion errors.

M04, normal test and the personal clips are evaluated only after development
selection. The final deployment decision additionally requires M04 and normal
retention plus either a 5% relative command-test WER improvement or a five-point
absolute gain in top-five exact coverage.

If these conditions fail, the result says `keep_v1`; the run still preserves
the best v3 research checkpoint and complete evidence.

## Output map

| File | Purpose |
| --- | --- |
| `result.json` | Complete baseline, history, selected checkpoint and protected tests |
| `next_decision.json` | Compact automatic deployment decision |
| `SHARE_THIS_WITH_CODEX.txt` | Exact content to paste here |
| `adapter.safetensors` | Selected v3 audio/projector plus decoder-LoRA adapter |
| `adapter_config.json` | Foundation identity, literal prompt and adapter metadata |
| `checkpoint-latest.pt` | Exact interruption-resume state |
| `checkpoint-best-candidate.pt` | Lowest development objective regardless of guards |
| `checkpoint-best-deployable.pt` | Best guarded state, when one exists |
| `predictions/*.tsv` | Per-utterance top-one outputs |
| `diagnostics/v1_top5.tsv` | Raw v1 alternatives on protected diagnostics |
| `diagnostics/v3_top5.tsv` | Raw v3 alternatives on protected diagnostics |
| `command_v3.log` | Complete live run log |

