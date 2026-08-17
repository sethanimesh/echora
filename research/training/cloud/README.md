# Qwen Gate 2 — L40S runbook

> Gate 2 is complete. For the full TORGO adaptation, use
> `research/training/PILOT-README.md`; this file is retained as the mechanics-probe
> record.

This is the copy-paste guide for the second paid gate. It does **not** perform
the longer TORGO fine-tune. It only proves that our selected Qwen3-ASR 1.7B
foundation can:

1. backpropagate through dysarthric audio on one L40S;
2. reduce loss on a fixed six-recording subset;
3. save a partial-training checkpoint;
4. reload that checkpoint and resume for one optimizer step.

Nothing here uses the Echora candidate ranker, personal context, the three
personal clips, data augmentation, CTC, or a different model.

## Frozen recipe

- Official model: `Qwen/Qwen3-ASR-1.7B-hf` at revision
  `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`.
- Exact foundation SHA-256:
  `2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1`.
- GPU: one L40S. The scripts refuse CPU, MPS, or another GPU name.
- M04 is the untouched outer-test speaker and F03 is the untouched development
  speaker. Neither is uploaded in the probe data.
- Trainable weights: Qwen audio-encoder layers 20–23 and the multimodal
  projector. The language decoder and the first 20 audio layers stay frozen.
- Six dysarthric recordings: one fixed recording from each remaining training
  speaker.
- Batch size: one. Eighteen optimizer steps, one checkpoint reload, then one
  resumed step.
- Learning rate: `5e-5`. No augmentation and no silent fallback.

The 179.9-second TORGO recording labelled only `hill` is deliberately excluded
by the frozen 30-second probe limit. That recording contains far more audio
than its one-word prompt can literally label, so it is not a trustworthy
supervised-training example. The longest eligible recording is still tested
with a forward/backward pass before the tiny overfit begins.

## 1. Make the upload bundle on the Mac

Open Terminal and start from the project root. This avoids the path mistake
that happened with the first gate:

    cd "/Users/animesh/Animesh/Project 2.0/echora"
    chmod +x research/training/cloud/prepare_qwen_gate2_bundle.sh
    research/training/cloud/prepare_qwen_gate2_bundle.sh

The final line must say:

    Created /private/tmp/echora-qwen-gate2.tar.gz

The archive contains only the seven fixed TORGO audio files, the frozen config,
and the Gate 2 scripts. It does not contain the personal clips or the 4 GB model.

## 2. Create the RunPod

Create one L40S Pod with SSH enabled and a persistent `/workspace` volume. Do
not select another GPU if L40S is unavailable. Copy the exact SSH command that
RunPod gives you; it will contain a new IP address and port.

No automatic Pod shutdown is installed. You remain in control of stopping it.

## 3. Upload and enter the Pod

Use this current RunPod endpoint from the SSH command you provided:

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 21605 \
      /private/tmp/echora-qwen-gate2.tar.gz root@195.26.232.139:/workspace/

    ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 root@195.26.232.139 -p 21605

Everything from the next section runs inside the RunPod SSH session.

## 4. Extract and set up the exact environment

    mkdir -p /workspace/echora
    tar -xzf /workspace/echora-qwen-gate2.tar.gz -C /workspace/echora
    chmod +x /workspace/echora/research/training/cloud/setup_qwen_gate2_l40s.sh
    chmod +x /workspace/echora/research/training/cloud/run_qwen_gate2.sh
    /workspace/echora/research/training/cloud/setup_qwen_gate2_l40s.sh \
      2>&1 | tee /workspace/echora/qwen_gate2_setup.log

The setup installs the pinned Transformers version, downloads the pinned
official Qwen foundation into `/workspace/models`, and verifies its byte count
and SHA-256. Success ends with:

    QWEN_GATE2_SETUP_DONE: exact Qwen foundation verified on NVIDIA L40S

If setup fails, do not install a different version or download another model.
Copy the final 100 log lines and paste them into this Codex task:

    tail -n 100 /workspace/echora/qwen_gate2_setup.log

## 5. Run the probe

Inside the RunPod:

    /workspace/echora/research/training/cloud/run_qwen_gate2.sh \
      2>&1 | tee /workspace/echora/qwen_gate2.log

The script has a lock, so a second terminal cannot accidentally launch another
paid copy. Leave this SSH window open; there is no need to poll it. The
completion hook prints exactly one of these markers:

    QWEN_GATE2_DONE: checkpoint reload and resume passed; copy /workspace/echora/checkpoints/qwen3-asr-1.7b-gate2-v1 before stopping the Pod

or:

    QWEN_GATE2_FAILED: paste the final 100 log lines; do not change the recipe

The success marker is printed only after finite loss, at least 1% relative loss
reduction, checkpoint saving, a fresh model reload, and one resumed optimizer
step. On failure, run this and paste the output here:

    tail -n 100 /workspace/echora/qwen_gate2.log

Do not rerun with changed layers, learning rate, model, GPU, or data.

## 6. Copy the results before stopping the Pod

After `QWEN_GATE2_DONE`, open a second Terminal window on the Mac:

    cd "/Users/animesh/Animesh/Project 2.0/echora"
    mkdir -p research/training/results/qwen3-asr-1.7b-gate2-v1
    mkdir -p "/Volumes/Extreme Pro/echora/configs/qwen3-asr-1.7b-gate2-v1"
    mkdir -p "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-gate2-v1"

Use the same endpoint for the data copy:

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 21605 \
      root@195.26.232.139:/workspace/echora/checkpoints/qwen3-asr-1.7b-gate2-v1/result.json \
      research/training/results/qwen3-asr-1.7b-gate2-v1/result.json

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 21605 \
      root@195.26.232.139:/workspace/echora/qwen_gate2.log \
      research/training/results/qwen3-asr-1.7b-gate2-v1/qwen_gate2.log

Copy the frozen configuration and both resumable checkpoints to the SSD:

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 21605 \
      root@195.26.232.139:/workspace/echora/checkpoints/qwen3-asr-1.7b-gate2-v1/qwen_gate2_probe.json \
      "/Volumes/Extreme Pro/echora/configs/qwen3-asr-1.7b-gate2-v1/"

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P 21605 \
      root@195.26.232.139:/workspace/echora/checkpoints/qwen3-asr-1.7b-gate2-v1/checkpoint-step-18.pt \
      root@195.26.232.139:/workspace/echora/checkpoints/qwen3-asr-1.7b-gate2-v1/checkpoint-resumed-step-19.pt \
      "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-gate2-v1/"

Confirm all four important files exist:

    ls -lh \
      research/training/results/qwen3-asr-1.7b-gate2-v1/result.json \
      "/Volumes/Extreme Pro/echora/configs/qwen3-asr-1.7b-gate2-v1/qwen_gate2_probe.json" \
      "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-gate2-v1/checkpoint-step-18.pt" \
      "/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-gate2-v1/checkpoint-resumed-step-19.pt"

Paste `result.json` into this Codex task. I will compare the initial and final
loss, memory use, runtime, trainable parameter count, and checkpoint hash before
we authorize a longer training pilot.

## 7. Stop the paid Pod

Once the four files above are confirmed on the Mac/SSD, stop or terminate the
RunPod yourself. The scripts never stop it automatically.

Passing Gate 2 does not prove better ASR. It only proves that the exact partial
tuning and checkpoint path work. The next paid pilot will require a separately
approved speaker-disjoint evaluation plan and ordinary-English retention data.
