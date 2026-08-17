# Echora

Echora is a local-first communication assistant for people whose speech is difficult to understand. It keeps literal ASR evidence visible, proposes a small number of message candidates, and waits for the speaker to confirm before communicating anything.

## Quick start on this Mac

The verified Qwen3-ASR foundation and tuned adapter are already stored under `models/echora-qwen3-asr-command-v3`.

```bash
./scripts/setup_local.sh
./scripts/dev.sh
```

Open <http://localhost:3000>. The backend API and interactive documentation are at <http://localhost:8000/docs>.

With the applications running, repeat the three-clip acceptance check in another terminal:

```bash
.venv/bin/python scripts/acceptance_local.py
```

The existing `.env` is preserved. `GROQ_API_KEY` is the preferred key name; the legacy `GROQ` variable also works. If no Groq key is configured, literal ASR continues to work and the UI displays all raw candidates.

## How the result is produced

1. The browser records or uploads an utterance.
2. The speaker chooses a session setting: General, Home, Hospital/care, or Outdoors.
3. The tuned Qwen model returns up to five immutable literal hypotheses.
4. Groq groups hypotheses into grounded intents using the setting as a weak prior. A key term survives only if a strict majority of the grouped beams carry it, counted both by search weight and by beam count.
5. A configurable evidence gate decides whether one intent is clear or multiple intents must be shown. An intent whose grouped beams disagree on competing content words is never cleared by a speech-act cue alone.
6. A second Groq pass turns each displayed intent into a natural communication message.
7. Deterministic grounding rejects wording that introduces unsupported substantive terms.
8. The speaker edits or selects a message and explicitly confirms it.
9. Only a confirmed message can be copied or spoken.

The displayed search weights are relative beam-search evidence, not calibrated confidence. Grammar repair never changes the stored literal transcript.

The clarity gate uses relative beam margin, normalized entropy, grouped intent weight, semantic evidence strength, and explicit speech-act cues. Defaults are configured with `ECHORA_CLEAR_MARGIN`, `ECHORA_CLEAR_MAX_ENTROPY`, `ECHORA_CLEAR_INTENT_WEIGHT`, and `ECHORA_CLEAR_MODERATE_INTENT_WEIGHT`. These are decision rules, not confidence probabilities.

## Cloud inference

- A persistent GPU Pod workflow is documented in `deploy/runpod/pod/README.md`.
- A scale-to-zero queue worker is documented in `deploy/runpod/serverless/README.md`.

Change `ECHORA_ASR_BACKEND` in `.env` to `pod` or `runpod` after configuring the corresponding variables. Backend selection is explicit; Echora never silently changes models or compute backends.

## Repository map

- `backend/` — FastAPI, local/remote ASR backends, Groq prompt chain, and tests.
- `frontend/` — accessible React communication interface.
- `models/` — verified, inference-only model bundle.
- `deploy/` — RunPod Pod and Serverless deployment packages.
- `research/` — historical benchmarks, training recipes, decisions, and tests.
- `data/` — local raw and derived datasets.
- `docs/` — architecture and operational details.

No personal context, accounts, database, or persistent message history are used in this version.

## Verify the repository

```bash
.venv/bin/pytest -q
npm --prefix frontend test
```
