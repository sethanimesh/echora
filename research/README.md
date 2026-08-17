# Echora research archive

This directory preserves the complete benchmark and training history that produced the deployable command-v3 adapter.

- `benchmarks/` contains baseline runners, personal clips, pause stress tests, evaluation utilities, and recorded decisions.
- `training/` contains the TORGO/Common Voice preparation code, augmentation, Gate 2, pilot, literal-v2, command-v3 recipes, cloud scripts, and unit tests.
- `notes/` contains the research roadmap and experiment gates.

Production inference does not import this package. The implementation in `backend/app/asr` is intentionally self-contained so training dependencies and paths cannot leak into the demo.
