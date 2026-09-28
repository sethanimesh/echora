# Reusable language tools

This package preserves the Hinglish CLI, provider adapters, Haystack components,
and optional Fish streaming/alignment/enrollment tools from the second project.
It uses the **same language core and lexicon** as the unified communication app:
`communication/backend/hinglish_core`. Compatibility imports preserve the
original `echora.*` API and class identity. `Pipeline.from_config` is a thin
composition adapter; it inherits the shared `run` implementation.

From the canonical project root, an offline example is:

```sh
.venv/bin/python -m echora.cli -c offline --json "Main kal office jaunga"
```

The installable `echora` entry point supports text, stdin, explanation tables,
JSON, optional Fish audio/timestamps, playback and explicitly authorized voice
enrollment. Provider dependencies are optional extras in `pyproject.toml`.

Library `Config` supports `offline`, `ollama`, `groq`, and `auto` (local model,
cloud fallback, then preserving heuristic). These are **library choices**; they
do not change the communication app's selected recognizer, wording model,
automatic speech policy, or revision-bound playback. `Config.from_env()` reads
process environment only. Use `--env-file PATH` or `Config.from_env(dotenv=...)`
when a particular environment file should be read.

Haystack's `build_pipeline(Config(classifier="offline"))` runs the shared stages
without a model call. `build_voice_pipeline(..., synthesizer=...)` adds a lazy
speech stream; synthesis starts when its iterator is consumed. Framework
adapters remain optional. The active app retains its async message lifecycle.

Validation:

```sh
.venv/bin/python -m pytest language_tests
```

Tests block network access, use mocked synthesis, and always skip the live Fish
smoke case. The Haystack test runs when the optional framework is installed.
`SOURCE.json` records the retained source hashes and consolidation decisions.
