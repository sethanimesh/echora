# L40S foundation gate handoff

Deployment addresses are user-supplied. Set `POD_HOST` and `POD_SSH_PORT` to your own Pod endpoint before running the SSH/SCP examples; the original endpoint is omitted from this public recipe.

This is the reproducible handoff for the first paid-GPU experiment. It compares
literal ASR from the official, untouched `nvidia/parakeet-tdt-1.1b` foundation
against the official, untouched `Qwen/Qwen3-ASR-1.7B-hf` foundation on the same
400 TORGO utterances. There is no candidate ranker, LLM correction, prompt, or
personal context in this gate.

## What is already done

- The fixed gate contains 400 utterances: 50 duration-stratified samples from
  each of TORGO's eight dysarthric speakers.
- Qwen3-ASR has completed all 400 locally. Its result is
  `research/benchmarks/results/foundation_gate/qwen3_asr_1_7b.tsv` (401 lines with header).
- The exact official foundation files on the SSD were SHA-256 verified.
- The cloud bundle was uploaded and extracted to `/workspace/echora`.
- The L40S environment has NeMo 2.7.3, torch 2.13.0+cu130, and working CUDA.
- The Parakeet download resumed from 68%, completed, and passed exact size and
  SHA-256 verification.
- The Parakeet gate completed and the clean 400-ID result was downloaded.
- Qwen is the provisional winner: 41.90% versus 45.83% speaker-macro WER, with
  lower deletion and empty-output rates.
- No personal recordings were uploaded.

RunPod preserves the volume disk mounted at `/workspace` when a Pod is stopped,
but not OS packages installed on the container disk. After this Pod restarted,
`ffmpeg` therefore had to be reinstalled even though the model, venv, gate, and
partial download remained available under `/workspace`.

Current RunPod SSH command:

    ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 root@${POD_HOST} -p ${POD_SSH_PORT}

## Continue from the current RunPod state

First, from the project root on the Mac, transfer the new run script (the
original uploaded bundle predates this handoff file):

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P ${POD_SSH_PORT} \
      research/benchmarks/cloud/run_parakeet_gate.sh \
      root@${POD_HOST}:/workspace/echora/research/benchmarks/run_parakeet_gate.sh

Then SSH into the machine and run:

    cd /workspace/echora
    chmod +x research/benchmarks/run_parakeet_gate.sh
    research/benchmarks/run_parakeet_gate.sh 2>&1 | tee /workspace/echora/parakeet_gate.log

The script resumes the official model download, checks its exact size and
SHA-256, runs a one-utterance CUDA smoke test, then completes the resumable
400-utterance gate. Success ends with:

    PARAKEET_GATE_DONE: /workspace/echora/research/benchmarks/results/foundation_gate/parakeet_tdt_1_1b.tsv has 400/400 utterances

Do not change the model, batch size, manifest, transcript normalization, or
decoding code during this run. If the command fails, paste the final 100 log
lines here instead of improvising a substitute:

    tail -n 100 /workspace/echora/parakeet_gate.log

## Copy the result to the Mac

Run this from the project root on the Mac:

    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P ${POD_SSH_PORT} \
      root@${POD_HOST}:/workspace/echora/research/benchmarks/results/foundation_gate/parakeet_tdt_1_1b.tsv \
      research/benchmarks/results/foundation_gate/parakeet_tdt_1_1b.tsv

Confirm that it has the header plus all 400 results:

    wc -l research/benchmarks/results/foundation_gate/parakeet_tdt_1_1b.tsv

The expected answer is `401`.

## Compare Parakeet with Qwen3-ASR

Still from the project root on the Mac:

    .venv/bin/python research/benchmarks/compare_foundations.py \
      research/benchmarks/results/foundation_gate/parakeet_tdt_1_1b.tsv \
      research/benchmarks/results/foundation_gate/qwen3_asr_1_7b.tsv \
      --json-output research/benchmarks/results/foundation_gate/comparison.json

Paste that JSON output here. The primary metric is speaker-macro WER so one
easier speaker cannot hide failure on another. Deletion rate, empty-output
rate, worst-speaker WER, and median real-time factor are also reported. A model
is only a provisional winner when its macro WER is at least 5% relatively lower
without worsening deletion or empty-output rate by more than two absolute
percentage points. Otherwise the script explicitly requests the full
evaluation instead of manufacturing a winner.

## Restart from zero on another L40S

The commands below rebuild the exact same portable bundle. Only the selected
400 TORGO recordings and the literal-ASR benchmark code are included.

On the Mac, from the project root:

    chmod +x research/benchmarks/cloud/prepare_bundle.sh
    research/benchmarks/cloud/prepare_bundle.sh
    scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -P <PORT> \
      /private/tmp/echora-cloud-gate.tar.gz root@<IP>:/workspace/

On the new L40S:

    mkdir -p /workspace/echora
    tar -xzf /workspace/echora-cloud-gate.tar.gz -C /workspace/echora
    chmod +x /workspace/echora/research/benchmarks/cloud/setup_l40s.sh
    chmod +x /workspace/echora/research/benchmarks/cloud/run_parakeet_gate.sh
    /workspace/echora/research/benchmarks/cloud/setup_l40s.sh
    /workspace/echora/research/benchmarks/cloud/run_parakeet_gate.sh 2>&1 | tee /workspace/echora/parakeet_gate.log

## What happens after the comparison

Do not start fine-tuning until the comparison is reviewed. The next plan is:

1. Select the foundation using the gate above, then confirm it on the complete
   speaker-independent TORGO evaluation split and the untouched personal pause
   suite. TORGO speakers must remain disjoint between training and evaluation.
2. Fine-tune only the selected foundation, starting with parameter-efficient
   tuning. For Parakeet this means encoder adapters/LoRA while initially keeping
   most foundation weights frozen; for Qwen it means LoRA on the audio-language
   projection and selected upper layers. We will choose the exact modules only
   after the foundation result, not beforehand.
3. Train on clean TORGO plus controlled online augmentation: internal silence
   insertion, time stretching, limited speed perturbation, gain/noise/reverb,
   and SpecAugment. Preserve the transcript and never augment evaluation audio.
4. Add ordinary English speech during training if TORGO-only tuning damages
   general ASR. It is a retention set, not a replacement for atypical speech.
5. Save resumable checkpoints and every hyperparameter/config under
   `/Volumes/Extreme Pro/echora`; copy checkpoints off the cloud intermittently.
6. Stop early if speaker-macro WER and pause-suite deletion rate do not improve.
   A demo checkpoint must improve literal recognition, not merely produce more
   fluent guesses. Candidate ranking remains downstream and outside ASR metrics.

The 400-utterance gate chooses where to spend training money; it is not a claim
that TORGO alone yields a universal stroke-survivor ASR model. The later test
set must include unseen speakers, longer natural pauses, stretched phonemes,
and eventually consented speech from multiple stroke survivors.

## Cloud cleanup

Copy the TSV and log first. Then either terminate the RunPod, or remove only
the uploaded gate audio and archive:

    rm -rf /workspace/echora/gate
    rm -f /workspace/echora-cloud-gate.tar.gz

No automatic shutdown was installed. Stop the paid instance yourself when the
result has been copied.
