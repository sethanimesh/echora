# Echora Qwen3-ASR command-v3

This is the complete inference bundle selected by the command-v3 experiment. It contains the official Qwen3-ASR 1.7B foundation, the final epoch-7 adapter, its exact training/inference configuration, and the protected-test report.

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
