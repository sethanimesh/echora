# Literal ASR v2: Qwen top-5 plus character CTC

Deployment addresses are user-supplied. Set `POD_HOST` and `POD_SSH_PORT` to your own Pod endpoint before running the SSH/SCP examples; the original endpoint is omitted from this public recipe.

This is the exact next experiment. It starts from the verified v1 Qwen adapter,
trains a monotonic character-CTC branch, then permits a small joint update to
Qwen's top four audio layers and multimodal projector. The Qwen text decoder
remains frozen.

The two reported outputs are literal ASR hypotheses:

- `qwen_top1` and `qwen_hypotheses`: Qwen rank 1 and top 5.
- `ctc_transcript`: an independent monotonic acoustic transcript.

There is no candidate ranker, personal-context ranker, grammar correction, or
intent reconstruction in this experiment. “I water” is scored against exactly
“I water.” The top-5 oracle is diagnostic only and is never substituted for
rank 1 WER.

## What this run answers

1. Was the literal phrase present in Qwen's search but suppressed at rank 1?
2. Can a monotonic CTC branch preserve short or grammatically unusual words
   better than Qwen's autoregressive decoder?
3. Can joint acoustic tuning improve that branch without damaging the good v1
   Qwen adapter or ordinary English?

TORGO is useful for dysarthric acoustics and many short prompted utterances,
but it is not a broad corpus of spontaneous telegraphic speech. Therefore this
run is a high-value test, not proof that the universal-ASR problem is solved.

## Expected GPU use

Use one NVIDIA L40S with at least 24 GB GPU memory. The v1 pilot used about
13.6 GiB. This run should remain in the same general range because the CTC head
is small, although actual peak memory is measured and written to `result.json`.

Budget approximately 75–110 minutes after setup for both top-5 diagnostics,
the v1 baselines, one fixed-encoder CTC warm-up epoch, up to four joint epochs,
and final evaluation. Early stopping can shorten it. Setup/model download time
depends on the Pod cache and network. Do not terminate the Pod until the result
directory has been copied to the SSD.

## Step 1 — Build the upload archive on the Mac

Attach the SSD first. Run this exact command from any Mac directory:

```bash
cd "/path/to/echora"

./research/training/cloud/prepare_qwen_literal_v2_bundle.sh \
  /private/tmp/echora-qwen-literal-v2.tar.gz \
  "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors"
```

The builder verifies the v1 adapter SHA-256 before packaging anything. It also
stages all required TORGO, normal-speech, personal, and pause-stress data. It
does not modify the SSD adapter.

## Step 2 — Upload to a fresh L40S Pod

Use these exact commands for the current Pod:

```bash
scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P ${POD_SSH_PORT} \
  /private/tmp/echora-qwen-literal-v2.tar.gz \
  root@${POD_HOST}:/workspace/
```

Connect:

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -p ${POD_SSH_PORT} root@${POD_HOST}
```

## Step 3 — Extract and set up the Pod

Run on the Pod:

```bash
mkdir -p /workspace/echora
tar --no-same-owner -xzf /workspace/echora-qwen-literal-v2.tar.gz \
  -C /workspace/echora

chmod +x \
  /workspace/echora/research/training/cloud/setup_qwen_literal_v2_l40s.sh \
  /workspace/echora/research/training/cloud/run_qwen_literal_v2.sh

/workspace/echora/research/training/cloud/setup_qwen_literal_v2_l40s.sh
```

Messages about unknown `LIBARCHIVE.xattr.com.apple.*` headers are harmless Mac
metadata warnings. Continue if extraction exits successfully.

Setup must end with:

```text
LITERAL_V2_SETUP_DONE: exact environment and foundation verified on NVIDIA L40S
```

The script deliberately refuses a non-L40S GPU, a non-isolated environment,
or foundation weights with the wrong byte count or SHA-256.

## Step 4 — Start a new run

Run on the Pod:

```bash
/workspace/echora/research/training/cloud/run_qwen_literal_v2.sh fresh
```

Keep this terminal open. The runner writes live output to both the terminal and:

```text
/workspace/echora/checkpoints/qwen3-asr-1.7b-literal-ctc-v2/literal_v2.log
```

It also writes resumable state every 50 optimizer steps and at every epoch.
The final completion hook is this exact line:

```text
LITERAL_V2_DONE: copy ... and paste SHARE_THIS_WITH_CODEX.txt
```

No shutdown or Pod deletion command is installed or run.

## If SSH disconnects or training is interrupted

Reconnect and run:

```bash
/workspace/echora/research/training/cloud/run_qwen_literal_v2.sh resume
```

Use `resume` only for the same extracted bundle and run directory. `fresh`
refuses existing state and deletes nothing. `resume` refuses to start unless
`checkpoint-latest.pt` exists and matches the run, model, adapter, config, and
manifest identities.

If the run prints `LITERAL_V2_FAILED`, do not alter hyperparameters or packages.
Copy the final 150 lines:

```bash
tail -n 150 \
  /workspace/echora/checkpoints/qwen3-asr-1.7b-literal-ctc-v2/literal_v2.log
```

## Step 5 — Inspect the completion marker

Run on the Pod:

```bash
cat /workspace/echora/status/qwen3-asr-1.7b-literal-ctc-v2/DONE

cat /workspace/echora/checkpoints/qwen3-asr-1.7b-literal-ctc-v2/SHARE_THIS_WITH_CODEX.txt
```

Paste the full `SHARE_THIS_WITH_CODEX.txt` content into this task. It contains
the exact evidence needed for the next decision, including:

- v1 and v2 top-5 candidates for “I water”;
- whether CTC produced exactly “I water”;
- one-word and 2–3-word TORGO top-1, top-5 oracle, and CTC WER;
- v1-to-v2 Qwen TORGO-test and normal-test changes;
- the next decision branch and the experiment's limitations.

## Step 6 — Copy every result and checkpoint to the SSD

Run this on the Mac, not the Pod:

```bash
mkdir -p "/Volumes/Extreme Pro/echora/checkpoints"

scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P ${POD_SSH_PORT} -r \
  root@${POD_HOST}:/workspace/echora/checkpoints/qwen3-asr-1.7b-literal-ctc-v2 \
  "/Volumes/Extreme Pro/echora/checkpoints/"
```

Verify the important files on the Mac:

```bash
find "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-literal-ctc-v2" \
  -maxdepth 2 -type f -print | sort

shasum -a 256 \
  "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-literal-ctc-v2/adapter.safetensors" \
  "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-literal-ctc-v2/ctc_head.safetensors" \
  "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-literal-ctc-v2/checkpoint-best.pt" \
  "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-literal-ctc-v2/checkpoint-latest.pt"
```

Only after this copy and verification should the Pod be stopped or deleted.

## Output map

All paths below are inside the Pod run directory:

| File | Purpose |
| --- | --- |
| `result.json` | Full v1 baseline, selected development, final Qwen/CTC metrics, time, and memory |
| `next_decision.json` | Compact branch decision based only on literal-ASR evidence |
| `SHARE_THIS_WITH_CODEX.txt` | Exact content to paste here |
| `adapter.safetensors` | Selected Qwen audio-layer/projector adapter |
| `ctc_head.safetensors` | Selected 39-symbol literal CTC head |
| `checkpoint-latest.pt` | Interruption-resumable latest state |
| `checkpoint-best.pt` | Selected guarded state |
| `diagnostics/v1_top5.tsv` | v1 top-5 literal hypotheses before v2 training |
| `diagnostics/v2_top5_and_ctc.tsv` | v2 top-5 plus CTC transcript per utterance |
| `diagnostics/*.summary.json` | Overall and per-slice top-1/oracle/CTC metrics |
| `predictions/*.tsv` | Per-utterance dev/test/personal Qwen and CTC outputs |
| `literal_v2.log` | Complete live run log |

## How checkpoint selection works

The first selectable checkpoint is produced after CTC head warm-up while all
v1 Qwen weights are fixed. Therefore a valid v1-Qwen-plus-CTC artifact exists
before joint tuning.

A joint checkpoint can replace it only if all four guards pass relative to v1:

- Qwen TORGO-development WER degradation is no more than 0.02 absolute;
- Qwen normal-development WER degradation is no more than 0.02 absolute;
- Qwen TORGO-development deletion-rate increase is no more than 0.02;
- Qwen TORGO-development empty-output-rate increase is no more than 0.02.

Among guarded checkpoints, the lowest CTC TORGO-development WER is selected,
with CTC CER as the tie-breaker. Test data is reported after selection and is
not used to choose an epoch.

## Decision rules after the run

- If CTC gives exactly “I water” and does not trail Qwen on the 216-item
  one-word dysarthric slice, keep the dual literal outputs and next test them
  on more held-out speakers and real telegraphic speech. A personal-only CTC
  success is recorded separately and is not called universal success.
- If Qwen top 5 contains “I water” but rank 1 does not, the acoustic evidence is
  reaching search and decoder ranking is the immediate failure. Surface it as
  an ASR alternative; do not grammar-correct rank 1 for evaluation.
- If neither Qwen top 5 nor CTC contains “I water,” more TORGO epochs are not a
  justified default. The next data requirement is real, literal, short and
  telegraphic dysarthric/stroke speech with speaker-disjoint evaluation.
- If joint tuning worsens Qwen retention, deploy the fixed-v1 checkpoint plus
  CTC rather than the joint Qwen adapter.
