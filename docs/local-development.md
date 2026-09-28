# Local development and configuration

One local communication app for speaking, typing, and choosing phrases. This repository combines the adapted speech recognizer and native client with the multilingual communication, delivery, and accessibility flows from the second project.

## Run on this Mac

```sh
./scripts/dev.sh
```

Open http://localhost:3000. The API runs on http://localhost:8000. From the enclosing workspace, `./start.sh` starts the same app. The existing environment, model files, and provider configuration are ready; a fresh installation uses `./scripts/setup_local.sh` first.

The app no longer depends on the sibling `Echora 2.0/` directory. That directory and the workspace's `unification-baseline-2026-09-28/` preserve the original sources.

## Shared behavior

- A resolved completed suggestion speaks automatically. Tapping an alternative speaks those displayed words immediately. All literal alternatives remain available; one surviving generated suggestion is not a confidence guarantee.
- Unresolved questions and recognition failures do not speak automatically. A literal alternative can still be selected and spoken. Edited text speaks when requested.
- Stop, edits, a new message, or a new revision invalidate pending playback. Refreshing or reconnecting does not replay an old result.
- Short follow-ups such as “without sugar” use the last approved message, in the same context, for at most ten minutes. This visible reference can be forgotten and is held in session memory. A separate **Remember this message** action explicitly saves wording to the selected personal profile; speech never saves it automatically.
- Literal recognition evidence remains separate from suggested wording. Relative beam weights are search evidence, not calibrated confidence.

## What is available

The web app accepts recordings, audio uploads, typed text, and quick phrases. Choose adapted Qwen recognition or Groq Whisper explicitly. Adapted recognition supports the local tuned model, Pod, or RunPod; its hypotheses go through one composition chain, without a second rewrite.

Profiles combine known words, approved details, audiences and speaking styles with protected names, language choices, use/ask rules, and saved exact-message cues. Named places and optional location matching supply context. Explicit current words and clarification answers take precedence over usual preferences.

Local adapted English recognition also has an acoustic verification and bounded personal-retrieval route. It preserves Qwen's raw evidence, ranks complete literal hypotheses, and constrains the single wording pass to those readings. Missing, mismatched or insufficiently validated scorer artifacts request a choice. Other recognition routes remain explicitly labelled as using the existing selection behavior. See [verification and memory](verification-and-memory.md) for training, artifact gates, limitations and reproducible evaluation.

The web app retains English and Hindi/Hinglish wording and pronunciation, device and Fish speech, tone and pace controls, optional voice/facial delivery suggestions, dwell/switch/head controls, and experimental gaze boards. Delivery suggestions are separate from message meaning. Camera features require browser permission and deliberate activation.

The Expo client uses the same session, message revision, selection, confirmation, and temporary-reference workflow. It retains the adapted recognizer and Groq/device speech. The web camera, gaze, and Fish controls are not yet native UI features. See [mobile/README.md](../mobile/README.md).

## Configuration and data

All providers read the root `.env`; `.env.example` lists supported options. Groq enables wording and Whisper. Fish needs a key and reference voice. Gemini enables optional voice/facial suggestions. Missing providers leave manual controls and the user's text available; recognition never silently switches providers.

`ECHORA_SPEECH_AUTOPLAY=false` disables automatic arrival speech; explicit choice and Speak remain available. `ECHORA_PERSONAL_ENABLED=false` disables use of profiles for composition. Named places are independent of that switch.

Profiles, explicitly remembered wording, and rebuildable profile-scoped vectors live in the gitignored `data/personal/profiles.sqlite3`. Both clients offer individual memory deletion, clear and personal-profile deletion. Named places, including tagged coordinates, are stored by the local API in `data/personal/settings.json`; location matching runs in the client. Recordings, message jobs, and the short conversation reference remain ephemeral. Remote recognition and selected cloud wording, speech, or delivery providers receive the input required for that feature; this is a local-first app, not an entirely offline app.

Profile migration is additive, uses separate source identifiers, and never overwrites source stores or existing imported edits:

```sh
.venv/bin/python scripts/import_profiles.py
.venv/bin/python scripts/import_profiles.py --apply
```

The first command is a dry run. Six original profiles have already been imported on this Mac. The second project's built-in sample profiles remain available.

## Optional experiments

The gaze environment and existing face model assets are preserved locally. To start the additional loopback gaze service:

```sh
ECHORA_GAZE_ENABLED=true ./scripts/dev.sh
```

See [GazeFollower](../communication/experiments/gazefollower/README.md) and [facial cues](../communication/experiments/facial-cues/README.md). These are experiments; automated checks do not establish gaze accuracy or suitability for an individual speaker.

## Verification

```sh
.venv/bin/python scripts/check_unified.py research/benchmarks/tests research/training/tests
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run build
npm --prefix mobile test
npm --prefix mobile run export:ios
```

The Python check uses temporary profile storage, dummy credentials, and blocks network connections. With the app running, `.venv/bin/python scripts/smoke_unified.py` checks the web proxy and a synthetic typed-message lifecycle without calling speech or model providers. See [pipeline evaluation](pipeline-evaluation.md) for current checks and benchmark artifacts, and [the earlier unification verification](unification-verification.md) for the original merge's device observations.

## Repository map

- `backend/`: recognition runtime, grounding, places, native bridge, and compatibility API.
- `communication/backend/`: shared communication lifecycle, profiles, pronunciation, and optional providers.
- `frontend/`: primary web app, served on port 3000.
- `mobile/`: Expo client for the shared backend.
- `echora/`: preserved pronunciation library, CLI, and optional Haystack adapters, using the shared language core.
- `models/`, `research/`, `deploy/`: tuned model, experiments/benchmarks, and remote recognition workers.

See [architecture](architecture.md) for the current contracts.
