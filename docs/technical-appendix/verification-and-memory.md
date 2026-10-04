# Acoustic verification and explicit personal memory

The unified clients preserve literal ASR evidence, learned ranking, generated wording and authorization to speak as separate data. Immediate speech remains the user's selected interaction: a resolved, completed recommendation may speak on arrival, and an explicit alternative tap speaks its exact displayed words. The internal confirmation endpoint authorizes only the current text, pronunciation, delivery and revision. It does not grant permission to save history.

## Recognition and artifact gates

The local adapted English route freezes command-v3 Qwen and uses its ordered 2048-dimensional audio features. A separate small Transformer scores complete audio–transcript pairs. It never sees profile information. A bounded fusion stage may use retrieved relevance only among acoustically supported hypotheses. The learned score cannot recover a literal absent from the beam set.

`ranking` includes the route, status, decision, selected hypothesis, ordered IDs, individual score components, artifact identity and reasons. `evidence.hypotheses` retains the original IDs, literal text, sequence scores and relative search weights. Those search weights are not correctness probabilities. A missing, corrupt or mismatched artifact cannot authorize learned arrival selection. The experiment freezes Qwen's inference device and precision; a differing live decoding route remains advisory. Other recognizers retain their explicitly labelled legacy behavior.

The learned route allows one composition pass over permitted complete readings. Generated-option count cannot resolve an acoustic ambiguity. Every literal remains selectable, including the raw form of a generated recommendation. When the selected expansion fails grounding, the application keeps the literal choices silent; it does not speak another surviving expansion automatically.

The fixed automatic-selection threshold is 0.95. Calibration requires at least ten correct and ten incorrect development prompt groups and coverage of at least one group where all beams are wrong. Only the winning reading receives a selected-correctness probability; the other component scores remain uncalibrated evidence. Activation additionally requires at least twenty accepted independent audit groups with no incorrect accepted group or harmful context flip. Uncertainty intervals must accompany these counts: this empirical gate is not a clinical or population accuracy guarantee. Failing the gate still permits ordering assistance and explicit choices. Audio longer than the frozen 45-second validation scope remains available for ranking and clarification within the configured upload duration limit (60 seconds by default), but cannot receive learned automatic selection.

## Per-speaker memory

SQLite remains the source of truth. Additive tables store explicit remembered messages, versioned profile sources and chunked embeddings with foreign-key ownership. The original migration stores are preserved. Profile edits keep their own revision; memory changes have a separate revision.

The web and native interfaces offer **Remember this message**, individual deletion, clear and personal-profile deletion. Remember derives its target profile and exact displayed words from the current settled job. It is idempotent for a job revision and never calls synthesis. Choosing an alternative, internal confirmation, playback and session restoration do not save anything automatically. No recording bytes are retained in memory records, and approved wording is marked as unverified literal evidence.

Both transports expose the same operations under `/api` and `/api/v1/communication`:

| Operation | Route | Body |
| --- | --- | --- |
| Remember | `POST /messages/{id}/remember` | `revision`, `memory_revision` |
| List | `GET /profiles/{id}/memories` | — |
| Delete one | `POST /profiles/{id}/memories/{memory_id}/delete` | `memory_revision` |
| Clear | `POST /profiles/{id}/memories/clear` | `memory_revision` |
| Delete personal profile | `POST /profiles/{id}/delete` | `profile_revision`, `memory_revision` |

The English MiniLM encoder runs locally on CPU. It embeds every token through bounded overlapping chunks and searches only the selected profile's eligible sources. Recipient, scenario, place, audience and declared source scopes constrain retrieval. At most five distinct references are returned per literal candidate. Recency uses a thirty-day half-life for remembered messages; remembered wording older than 180 days contributes zero and remains available to manage or delete. Repeated records do not accumulate votes. Conservative lexical guards exclude conflicting negation, quantities, opposite directions and declared names. These guards do not prove unrestricted semantic consistency. Semantic references are not instructions or evidence that their facts were spoken. Only explicitly approved, acoustically anchored details can add substantive wording.

Source IDs, content hashes and revisions travel with the job. Deletion or relevant profile edits revoke pending derived wording and audio, and late provider completions cannot reinstate them. An additive table stores only the opaque job ID and displayed revision of deleted saves, preventing delayed Remember requests from recreating deleted content even with a freshly fetched memory revision. A newly reviewed displayed revision can be explicitly remembered again. Deleting a profile also deletes these revocation records. Explicit user edits remain available. The existing ten-minute follow-up reference remains temporary and separately forgettable.

Profile sources require declared English before semantic indexing; `original` and Hindi/Hinglish profiles retain their existing exact-profile behavior. English adapted messages explicitly remembered from those profiles can carry their actual recognized language. Non-English saved wording can be listed/deleted but is not silently translated into this index.

## Reproducible local experiment

Run in the canonical repository with the existing virtual environment:

```sh
.venv/bin/python research/training/acoustic_verification.py prepare
.venv/bin/python research/training/acoustic_verification.py benchmark
.venv/bin/python research/training/acoustic_verification.py cache
.venv/bin/python research/training/acoustic_verification.py train
.venv/bin/python research/training/acoustic_verification.py fit
.venv/bin/python research/training/acoustic_verification.py evaluate
```

`all` runs those stages in order; completed cache records and training checkpoints support resume. `--run-dir` selects another experiment directory. The default is `data/derived/verification/run-v1`; its `experiment.json` freezes corpus-relative remapping, memberships, model/decoding identity and configuration. Original files remain untouched. Duration filtering excludes entire out-of-contract recordings and is reported; it never cuts audio to fit. Use MPS on this Mac; no automatic remote compute fallback exists.

F03 is partitioned by normalized prompt group into checkpoint selection, fusion fitting, probability calibration and a final untouched acceptance audit. Repeated recordings and every context variant follow their group. M04 remains outside all fitting and is decoded only after the artifact is frozen. It is a historical speaker holdout already evaluated by the original project, not a newly untouched test set. Original prompt overlap is retained and disclosed. This is not an eight-fold independent model study.

All four transcription comparisons reuse identical decoded candidates: ASR rank one, acoustic reranking, ASR plus context, and full fusion. Frozen artificial context entries originate in training data, not individual test answers. Empty, relevant, wrong-recipient, stale, contradictory and misleading conditions are reported separately. No personal database is used for this research run.

The completed v1 experiment is **advisory only**: its audit accepted 14 independent groups, below the required 20, and both fitted contextual weights are zero. M04's small WER difference has an uncertainty interval spanning no improvement. The contradiction control also exposed a bounded-filter gap: opposing `no` / `not no` sources were retrieved in five M04 cases, with zero fusion contribution. This limitation remains in the frozen method and was not repaired by tuning against the evaluated test. See the [results and limitations](pipeline-evaluation.md) before interpreting the artifact or enabling a future one.

Run the separate wording evaluation with the configured Groq provider:

```sh
.venv/bin/python research/benchmarks/message_chain/evaluate_expansion.py
```

It uses frozen synthetic fixtures through the production composition boundary, repeats both plain and retrieved-context conditions, and saves resumable trials, a summary and blinded review CSVs under `data/derived/verification/expansion-v2`. The earlier chain-only v1 run remains preserved with its documented grading limitations. Provider failures and withheld wording are reported separately. Minimum reference edit counts are correction-effort proxies; actual effort and unrestricted meaning preservation require human review. No audio, camera, gaze accuracy or clinical benefit is established by automated checks.

Run `scripts/check_unified.py`, the research test suites, client tests/typechecks, the web build and native export after changes. Restart the backend to load a newly completed verifier artifact; do not manually mark an unsuccessful audit as passed.
