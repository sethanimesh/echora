# Changelog

## 2026-09-28 — Command-v3 Adapter Release

**Echora Qwen3-ASR Command-v3** — Literal ASR adapter for composed dysarthric commands.

### Highlights
- **29% WER reduction** on composed commands (72.58% → 51.58%) vs previous adapter
- **36.5% top-5 exact coverage** — +20.4 pts over v1, preserving literal alternatives
- **Normal speech retention** maintained at 5.23% WER
- **Five-beam literal output** with immutable evidence for downstream selection

### Adapter Specifications
- Base: `Qwen/Qwen3-ASR-1.7B-hf` (pinned revision `bcd2b5b7f3`)
- Tuned: audio layers 20–23, multimodal projector, rank-8 LoRA on decoder layers 20–27
- Trainable: 53.99M params (2.65%) · Epoch 7 of 8 · Seed 20260818
- Training: 6,000 composed commands (6 TORGO speakers) + Common Voice retention

### Evaluation
- Protected speaker holdout: M04 (480 composed commands, 281 TORGO, 262 normal)
- Speaker-disjoint folds, equal gap/stretch coverage
- Full reproducibility: `python3 scripts/reproduce_results.py`

### Known Limitations
- 8 dysarthric speakers total (TORGO) — not a population study
- Composed commands are controlled augmentations
- M04 shares 227/229 prompt groups with training
- Personal clip regression documented (3 utterances, target user)

### Technical Appendix
Pipeline experiments (learned verifier, context fusion, acceptance audit) are documented in [`docs/technical-appendix/`](docs/technical-appendix/) for research exploration.

---

## 2026-09-28 — Unified Application & Documentation

- Consolidated web and Expo clients around one backend message, revision, speech and explicit-memory lifecycle
- Preserved literal recognition alternatives and separated learned verification from generated-candidate count
- Added portfolio navigation, methodology, retrospective architecture decisions, comparative results, and reproduction instructions
- Added standard-library command to recompute foundation metrics from checked-in predictions, CI checks, and real interface captures using synthetic input

The application integration and experiments preceded the repository-presentation pass. See [development history](docs/development-history.md) and [pipeline evaluation](docs/technical-appendix/pipeline-evaluation.md) for their evidence and limitations. Earlier work remains recorded in Git rather than assigned invented release dates.