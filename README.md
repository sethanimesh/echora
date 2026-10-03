# Echora: Assistive Acoustic Model for Pathological Speech (2025–2026)

> ⚠️ **Research Prototype — Not a Clinical Tool**
> This is a personal research prototype exploring communication support for dysarthric speech.
> It has **not** been validated for clinical use, accessibility suitability, or individual benefit.
> See [Limitations](docs/limitations.md) and [Failure Analysis](docs/failure-analysis.md) before any use.

A research prototype exploring how to preserve literal speech recognition evidence while helping a speaker select a message. **Does not establish clinical benefit, accessibility suitability, or reliable recognition for any individual.**

## Why this is difficult

- **The right words may be missing.** Multiple decoder hypotheses expose alternatives but cannot recover a phrase absent from every beam.
- **Fluency can hide mistakes.** Generated wording must retain literal sources and meaningful alternatives, including negation, names, and quantities.
- **Context can contradict current speech.** Habitual preferences are weak, scoped evidence; explicit current words take precedence.
- **Late responses can speak an old message.** Edits, Stop, new work, and context changes must revoke pending playback across both clients.
- **Small datasets limit conclusions.** Speaker holdouts, repeated prompts, synthetic commands, and ordinary-speech retention measure different things.

## Evidence at a glance

These are separate experiments, not one leaderboard. WER is word error rate; lower is better.

| Experiment | Data Type | Baseline | Result | Interpretation |
| --- | --- | --- | --- | --- |
| Foundation screen, 400 utterances / eight speakers | Development (selection) | Parakeet: 45.83% speaker-macro WER | Qwen: 41.90% | Supported foundation selection; not final validation |
| Command-v3 protected composed-command test | Held-out test (M04) | Previous adapter: 72.58% WER | v3: 51.58% | Controlled compositions, not naturally spoken commands |
| Command-v3 normal-speech retention | Held-out test | 5.23% WER | 5.23% | Retention on this test |
| Three personal recordings | Personal clips (target user) | Previous adapter: 22.22% WER | v3: 44.44% | A real regression |
| Later verifier, M04 | Held-out test (fusion vs ASR) | ASR: 52.62% WER | Fusion: 52.33% | Paired interval crosses zero; improvement not established |

The separate five-beam command diagnostic finds the exact reference somewhere in the beams for **36.46%** of utterances. This is oracle coverage, not automatic-selection accuracy. The learned verifier's acceptance audit failed its minimum evidence requirement, so **learned automatic selection remains disabled**. [Sources and protocol differences →](docs/evaluation.md)

## Critical Limitations at a Glance

| Limitation | Detail |
|------------|--------|
| **Speakers** | Only **8 dysarthric speakers** (TORGO) across all train/dev/test folds |
| **Task** | Results on **composed commands** (stitched isolated words), not natural speech |
| **Personal regression** | Released adapter **worsened** from 22.22% → 44.44% WER on 3 target-user clips |
| **Learned selection** | **Disabled** — acceptance audit failed (14/20 required groups); all results request user choice |
| **Test overlap** | Held-out speaker M04 shares 227/229 prompt groups with training |
| **No clinical validation** | No participant study, IRB, or SLP assessment |

## Interface

![Echora connected interface](assets/screenshots/main-interface.png)

*Screenshot of running application using **synthetic typed input** (not speech recognition), isolated demonstration profile store, and all cloud provider keys disabled. [Demo walkthrough](docs/demo.md) — does not demonstrate recognition quality or audible playback.*

## How it works

```mermaid
flowchart LR
    Input[Speech, typing, or phrase] --> Evidence[Literal evidence with provenance]
    Evidence --> Decision[Verification or labelled legacy decision]
    Profile[Scoped profile context] --> Decision
    Evidence --> Compose[One grounded wording pass]
    Decision --> Compose
    Compose --> UI[Editable message and alternatives]
    Evidence --> UI
    UI --> Revision[Current revision speech authorization]
    Revision --> Voice[Device or selected provider speech]
    UI --> Remember[Explicit Remember action]
    Remember --> Profile
```

The local adapted English route separates acoustic verification from generated-candidate count. Missing or insufficiently validated verifier artifacts request a choice. A resolved completed suggestion may speak automatically; choosing an alternative speaks those exact displayed words. Internal confirmation binds playback to the current revision without adding a user step. Stop, edits, and new work invalidate it; restoring a session never replays old speech.

All distinct literal beams remain selectable. A visible follow-up reference lasts at most ten minutes in the same context. Only **Remember this message** persists exact wording to the selected profile. Remembered wording is not acoustic training truth.

The web client includes Hindi/Hinglish wording, delivery controls, and optional experimental camera/gaze features. Expo shares the backend lifecycle with fewer optional controls. [Interaction and data boundaries →](docs/architecture.md)

## Inspect or run

Recompute the foundation comparison from checked-in predictions and summarize saved adapter results without models, keys, or audio:

```sh
git clone https://github.com/sethanimesh/echora.git
cd echora
python3 scripts/reproduce_results.py
```

This verifies **saved predictions only** — it does not rerun inference, retrain the adapter, or download models/audio. For the application, use an Apple Silicon Mac, Python 3.12–3.14, Node 22.13+, FFmpeg, and the verified model bundle:

```sh
cp .env.example .env
# Provision the model bundle and provider settings; see the guide below.
./scripts/setup_local.sh
./scripts/dev.sh
```

Open `http://localhost:3000`; the shared API runs on 8000. **Weights, recordings, and private profiles are excluded from Git.** Setup verifies the bundle but does not download it. [Complete prerequisites and reproduction boundaries →](docs/reproducibility.md)

[Demo walkthrough](docs/demo.md) covers typed input, editing, speech and Stop. Automated checks do not establish audible speech quality, microphone recognition, or gaze accuracy.

## Tests

```sh
.venv/bin/python scripts/check_unified.py research/benchmarks/tests research/training/tests
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run build
npm --prefix mobile test
```

The Python runner isolates personal storage and blocks external connections. CI is configured to run these contracts and client checks without provider credentials. [Testing scope →](docs/reproducibility.md#testing)

## Repository guide

| Area | Purpose |
| --- | --- |
| [backend/](backend/) | Recognition, grounding, verification and native bridge |
| [communication/backend/](communication/backend/) | Shared lifecycle, profiles, language and delivery |
| [frontend/](frontend/) / [mobile/](mobile/) | Web and Expo clients |
| [research/](research/) | Training, baselines, benchmarks and saved results |
| [models/](models/) | Artifact identities, configurations, checksums and reports |
| [docs/README.md](docs/README.md) | Methodology, decisions, failures and open questions |

Groq, Fish, optional Gemini, and remote recognition receive the inputs required by their selected features. Profiles use local SQLite. Tagged coordinates are stored through the local API and matched in the client. This is a local-first application with optional cloud providers. [Configuration and data →](docs/local-development.md#configuration-and-data)

See the [development record](docs/development-history.md) for the preserved chronology and [contributions](docs/contributions.md) for upstream components and attribution boundaries.

The public repository retains training code, evaluation predictions, model identities, design decisions, contract tests, and documented failures. Personal recordings, transcript sidecars, individual facial-trial exports, and local audit originals stay in ignored storage. Published personal smoke-test results use stable recording aliases; their literal wording and measurements are preserved. See [research evidence boundaries](research/README.md#public-evidence-and-private-inputs).