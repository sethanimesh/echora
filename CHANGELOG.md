# Changelog

Entries describe repository milestones, not a reconstructed release schedule.

## Current Status (as of 2026-09-28)

- **Learned automatic selection: DISABLED** — acceptance audit failed (14/20 required)
- **Personal regression documented:** v3 WER 44.44% vs v1 22.22% on 3 target-user clips
- **8 speakers total** across all folds — not a population study
- **Composed commands only** — controlled augmentations, not natural speech

## 2026-09-28 — Unified application and evidence documentation

- Consolidated web and Expo clients around one backend message, revision, speech and explicit-memory lifecycle.
- Preserved literal recognition alternatives and separated learned verification from generated-candidate count. The learned acceptance audit failed, so automatic selection on that route remains disabled.
- Added portfolio navigation, methodology, retrospective architecture decisions, comparative results, failure analysis, validity limits and reproduction instructions.
- Added a standard-library command to recompute foundation metrics from checked-in predictions, CI checks and real interface captures using synthetic input.

The application integration and experiments preceded the repository-presentation pass. See [development history](docs/development-history.md) and [pipeline evaluation](docs/pipeline-evaluation.md) for their evidence and limitations. Earlier work remains recorded in Git rather than assigned invented release dates.
