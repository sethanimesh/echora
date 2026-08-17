# Qwen TORGO pilot v1 — verified result

Status: **passed**. Selected checkpoint: epoch 5.

## Main results

- F03 development WER: 22.02% foundation to 10.37% adapted, a 52.90%
  relative improvement.
- M04 outer-test WER: 85.57% foundation to 56.29% adapted, a 34.21%
  relative improvement.
- M04 deletion rate: 5.37% foundation to 3.96% adapted.
- Normal-English test WER: 3.74% foundation to 5.23% adapted, a 1.48
  absolute-point degradation within the fixed 2-point guardrail.
- Empty outputs: zero on M04 and normal test.
- Personal clips: 33.33% foundation WER to 22.22% adapted WER.
- Peak allocated L40S memory: 13.56 GiB.
- Total run time: 4,900 seconds (81.7 minutes).

## Verified local artifacts

- `result.json`, `run_manifest.json`, `training_history.json` and the cloud log.
- Complete automatic `predictions/` directory. All ten prediction files named
  in `result.json` match their recorded SHA-256 values.
- SSD adapter SHA-256:
  `4e211db6c6571a7a4efbe8e851b67d44f33feea41724d0ab7afcf7195e018e0c`.
- SSD best-checkpoint SHA-256:
  `7c5078325e9fbac1136df10981a23f17411ddda306b611ee99e9ff526bc20fa9`.
- SSD frozen-config SHA-256:
  `698db922c8a86ab05deb39a9776ad789378008e622bf671a1fe75752de715a04`.
- Fresh foundation plus exported adapter reproduced the personal transcripts
  exactly.

The deployable adapter and best resumable optimizer checkpoint are therefore
safe locally.

## Artifacts not copied from the Pod

- `checkpoint-latest.pt`, expected SHA-256
  `8af6793d583b7f0773ae316c85f1c5b0838248361b9a30ccd14ced91de32afc3`.
  This is useful only to resume from the final early-stopping state; it is not
  needed for inference and the verified epoch-5 best checkpoint is available.
- The standalone `personal-adapted.tsv` and three `pause-stress-*.tsv` files.
  Their transcripts and metrics were preserved from the pasted Pod console in
  `manual_smoke_results.json`, but their original timing columns were not.

The last saved SSH endpoint refused connections during the recovery audit. If
the persistent volume is still available under a new endpoint, copy these five
files. Otherwise no deployable model or primary evaluation result is lost.

## Interpretation of the personal failure

The phrases “I want water” and “I want water please” remained correct after
adding 0.5, 1.0 and 2.0-second pauses. The short “I water” clip failed in every
condition (`gotten`, `high`, or `high button`). This isolates the remaining
problem more strongly to an acoustically ambiguous, telegraphic utterance plus
the decoder's language prior than to pause duration alone.
