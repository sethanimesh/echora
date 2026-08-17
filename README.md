# Echora

Echora is a local-first communication assistant for people whose speech is difficult to understand. It keeps literal ASR evidence visible, proposes a small number of message candidates, and speaks the one the speaker has settled on.

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
8. A single surviving message is spoken immediately with Groq TTS; when several remain, the speaker picks one and that tap speaks it.
9. The message stays editable, and an edited version is spoken on request. If Groq speech is unavailable the browser voice takes over, so a message is never left unsaid.

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

## Personal context

Choosing a profile lets Echora use what it knows about one speaker. It does two separate things, kept
apart because they carry different risk:

- A **disambiguation prior**: words from the speaker's life help choose between the variants the
  recognizer produced. `[marge|march|large]` resolves to Marge when Marge is their carer. This can
  never introduce a word no beam contained.
- An **anchored specialization**: the speaker's own version of an ordinary thing, so `coffee` becomes
  `Madras filter coffee`. This one adds words, so it is applied only when the profile declared that
  exact wording, the anchor survives the grounding check, and the anchor sits in a position every beam
  agreed on. It is marked in the interface and reverts in one tap.

Past accepted messages are retrieved as few-shot examples by a small local encoder. The query is the
share-weighted union of all five beams rather than the leading one, since the literal is in the top
five only about a third of the time. Storing a message merges it into the nearest existing entry, so
saying the same thing forty times leaves one entry with a count rather than forty rows.

`data/personas/` ships six demonstration speakers and is never written to; a profile is copied into
the gitignored `data/personal/` on first use. Fetch the encoder once with `./scripts/fetch_embedder.sh`,
or set `ECHORA_PERSONAL_ENABLED=false`. Whatever goes wrong in this layer, the message chain still runs.

No accounts or database are used, and nothing leaves the machine except the one Groq request.

## Verify the repository

```bash
.venv/bin/pytest -q
npm --prefix frontend test
```
