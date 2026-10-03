# Unified Echora — current implementation policy

These contracts describe the canonical application. README and docs/architecture.md explain the current behavior; research recipes record historical experiments.

- `echora/` is the canonical app. Root `scripts/dev.sh` starts the web client on 3000 and one shared backend on 8000. Never import code or environments from the sibling `Echora 2.0/` snapshot.
- A resolved completed suggestion speaks automatically; tapping an alternative, including a literal fallback, speaks the chosen words. Local adapted English recognition uses the separate verification decision when that route is active, not generated-candidate count. Missing/uncalibrated verification leaves this route asking for a choice. Other recognizers keep the labelled legacy route. `ECHORA_SPEECH_AUTOPLAY=false` suppresses arrival speech but not explicit choices.
- Confirmation is an internal, revision-bound speech authorization, not an extra user step. Edits, Stop and new work invalidate pending playback. Restoring or reconnecting never replays old suggestions.
- The last approved message may automatically supply a narrow follow-up for at most ten minutes within the same context. Keep its visible reference, Forget, profile/scope checks and expiry. Only an explicit Remember action persists the exact displayed message in the selected profile's retrieval store. Arrival, selection, confirmation and playback never save history automatically. Remembered wording is not verified acoustic training truth.
- Keep all literal beams selectable separately from generated messages. The verification scorer reads only frozen Qwen features and literal text; profile context enters bounded fusion afterward. Never alter raw scores or call them confidence. Scorer/calibration artifacts must pass identity and acceptance checks before enabling learned automatic selection.
- Preserve literal evidence, every meaningful alternative and honest provenance. There is one composition pass for adapted ASR. Do not silently switch recognition backends or truncate speech.
- Profiles use `data/personal/profiles.sqlite3`; migration is additive and preserves original source stores. Explicit current words and answers override habitual details. Native and web use the same lifecycle, even though their optional feature UIs differ.
- Tagged coordinates are saved through the local API in the places file; matching happens in the client. They do not travel to recognition or wording providers.
- Groq, Fish, and optional Gemini features send their required inputs to the selected provider. Do not describe the app as entirely offline or database-free.
- Use `scripts/check_unified.py` for isolated backend/language/native contract checks. See README and docs/architecture.md for current behavior. Do not claim microphone, camera, speech quality or gaze accuracy was tested from automated checks alone.

## Recognition and contextual wording contracts

- The deployed command-v3 adapter is selected at epoch 7. ASR returns up to five literal hypotheses with sequence scores and relative beam-search weights; these weights are not calibrated confidence.
- Session settings are `general`, `home`, `care`, and `outdoors`. A named place borrows one of these settings. Listener declarations are nullable: resolve place declaration, then profile preference for the setting, then the setting default, in `schemas.resolve_listener`. Preserve abstention throughout API/client normalization.
- Familiar listeners receive stated needs; unfamiliar listeners receive location questions or the complete service request. Preserve source hypothesis IDs and substantive key terms in generated wording.
- Personal priors may rank only words present in recognition evidence. Explicitly declared specialization requires a grounded anchor and the configured spelling-family share. Scope restricts offered details while the complete audit vocabulary continues to reject unsupported wording.
- Preserve demonstration baselines in `data/personas/`. Current profile edits and remembered messages belong in the selected profile's private SQLite store.
- Keep protected evaluation separate from selection. Retain failures, split definitions, raw scores, seeds, model identities, and reproducible public evidence. Private recordings, user reports, machine addresses, and agent session artifacts belong outside tracked files.
