# Reproducibility

## Inspect saved results without model access

From the repository root, run:

```sh
python3 scripts/reproduce_results.py
```

Python 3.12+ is sufficient; no third-party packages, provider keys, audio or GPU are required. The script recomputes foundation metrics from checked-in TSV predictions, checks them against the saved comparison, and prints the command-v3 report from its saved JSON. Expected runtime is seconds. It does not rerun inference, retrain the adapter, or independently regenerate command-v3 predictions.

## Run the application

The supported local model environment is an Apple Silicon Mac. Use Python 3.12–3.14, Node 22.13+, npm and FFmpeg. The original training environment was one NVIDIA L40S; training requirements are separate from local inference. Peak application memory requirements have not been established as a portable minimum.

1. Clone the repository and copy `.env.example` to `.env`.
2. Provision the model bundle below. It is intentionally excluded from Git.
3. Set provider credentials for the desired features. Groq supplies wording and optional recognition/speech; Fish and Gemini are optional. Keep keys in `.env`.
4. Run `./scripts/setup_local.sh`, then `./scripts/dev.sh`.
5. Open `http://localhost:3000`. The shared API is at `http://localhost:8000/docs`.

The setup script installs Python/web dependencies, migrates demonstration profiles additively and verifies model checksums. It does **not** download the foundation or adapter. It currently configures the repository's existing Git hook; see [contributions](contributions.md) for attribution boundaries.

### Model bundle

The [manifest](../models/echora-qwen3-asr-command-v3/manifest.json) pins:

- Foundation: `Qwen/Qwen3-ASR-1.7B-hf`, revision `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`.
- Adapter: `sethanimesh/echora-qwen3-asr-command`, revision `v3`, with its expected SHA-256.
- Selected epoch seven, five literal beams, and the exact literal prompt.

Place the foundation snapshot under `models/echora-qwen3-asr-command-v3/foundation/` and the adapter at `adapter/adapter.safetensors` inside that bundle. Use the existing adapter configuration and evaluation configuration. Obtain access separately if the model repository is private. Verify from the bundle directory:

```sh
shasum -a 256 -c CHECKSUMS.sha256
```

A missing or mismatched bundle is a setup failure, not permission to substitute another recognizer. Remote adapted recognition is an explicit alternative described under `deploy/`. A missing verifier artifact keeps learned selection in clarification mode.

### Profiles and optional features

Personal data lives under ignored `data/personal/`; shipped demonstration profiles are separate. [Local development](local-development.md) covers configuration, migration and optional gaze services. Expo setup is in [mobile/README.md](../mobile/README.md). Optional model assets and environments must be provisioned separately.

## Testing

```sh
.venv/bin/python scripts/check_unified.py research/benchmarks/tests research/training/tests
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run build
npm --prefix mobile ci
npm --prefix mobile test
```

The Python runner uses temporary profile storage, dummy credentials and blocked outbound sockets. Client tests cover state, selection, cancellation, memory and delivery behavior. Type checks and a production build catch interface/build errors. CI is configured for Linux/Python 3.12 and Node 22; a hosted CI pass must be observed separately from local validation.

The 28 September repository-preparation run passed **771 Python tests (two skipped), 77 web tests and 26 native tests**, both client type checks and the web production build. Existing framework/deprecation and large-bundle warnings remain. These are contract checks, not microphone, voice-quality, camera or gaze measurements.

## Retraining and later experiments

The [command-v3 recipe](../research/training/COMMAND-V3-README.md) records the original single-L40S experiment, frozen configuration, seed, splits and resume rules. Its machine paths and historical cost estimate are not portable defaults. Reproduction requires corpus access and regenerated manifests/audio; protected tests must remain separate from development selection.

[Verification and memory](verification-and-memory.md) records later experiment commands. Their detailed outputs remain local under ignored `data/derived/verification/`. Do not present the saved narrative as fully redistributable raw evidence. Human wording ratings and a new independent acceptance evaluation remain outstanding.
