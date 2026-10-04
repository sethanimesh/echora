# Echora Qwen3-ASR Command-v3

This directory defines the inference bundle for the command-v3 adapter. Git includes identity metadata, checksums, training/inference configuration and the protected-test report. The official Qwen3-ASR 1.7B foundation and epoch-7 adapter weights must be provisioned separately; see [reproducibility](../../docs/reproducibility.md#model-bundle).

## Contents

- `foundation/` — verified public Qwen foundation files.
- `adapter/adapter.safetensors` — 206 MB tuned audio/projector/LoRA state.
- `adapter/adapter_config.json` — artifact identity, prompt, and hashes.
- `evaluation/qwen_command_v3.json` — frozen architecture and decoding configuration.
- `evaluation/next_decision.json` — deployment decision and headline metrics.
- `evaluation/result.json` — complete experiment result.
- `manifest.json` and `CHECKSUMS.sha256` — production identity contract.

## Verify

```bash
cd models/echora-qwen3-asr-command-v3
shasum -a 256 -c CHECKSUMS.sha256
```

## Use

The supported entry point is the FastAPI application:

```bash
./scripts/dev.sh
```

For direct Python use, instantiate `backend.app.asr.engine.QwenCommandEngine` with `foundation/`, `adapter/adapter.safetensors`, and `evaluation/qwen_command_v3.json`.

The output is literal top-k ASR. Search weights are relative within a beam set and are not confidence probabilities. Message ranking and grammar correction belong to the application layer, not this model.

## Results Summary

| Metric | v1 (Previous) | v3 (Command-v3) | Change |
|--------|--------------|-----------------|--------|
| Composed commands WER (M04) | 72.58% | **51.58%** | **−21.00 pts** |
| Top-5 exact coverage | 16.04% | **36.46%** | **+20.42 pts** |
| Rank-1 WER (5-beam diag) | 77.58% | **61.75%** | −15.83 pts |
| Normal speech WER | 5.23% | 5.23% | unchanged |

Selected epoch: 7 of 8. Training updates acoustic layers, the multimodal projector, and decoder LoRA from the pinned foundation. F03 supplies development data, M04 the original protected speaker test. Composed commands use 6,000 training and 480 development/test examples each, with speaker- and composed-phrase-disjoint partitions.

See [MODELCARD.md](MODELCARD.md) for full details, limitations, and reproducibility instructions.