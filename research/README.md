# Echora research archive

This directory preserves the public benchmark and training evidence that produced the deployable command-v3 adapter.

- `benchmarks/` contains baseline runners, pause stress tests, evaluation utilities, and recorded decisions. User recordings and their transcript sidecars stay in ignored local storage.
- `training/` contains the TORGO/Common Voice preparation code, augmentation, Gate 2, pilot, literal-v2, command-v3 recipes, cloud scripts, and unit tests.
- `notes/` contains the research roadmap and experiment gates.

Production inference does not import this package. The implementation in `backend/app/asr` is intentionally self-contained so training dependencies and paths cannot leak into the demo.

## Public evidence and private inputs

Checked-in predictions retain literal references, recognizer outputs, scores, measured timings, split membership, and failures. Local home-directory prefixes are replaced by corpus-relative paths; three personal smoke-test recordings use stable `sample-01`, `sample-02`, and `sample-03` aliases in published results. These are aliases for real observations, not synthetic replacements. The original recordings and transcript sidecars are private prerequisites for rerunning those observations.

Personal trial exports and original files from repository cleanup are kept in ignored private storage. Historical recipes use user-supplied Pod addresses. Model identities, seeds, configurations, hashes, protected-test results, and reproducibility scripts remain public. `python3 scripts/reproduce_results.py` recomputes the saved foundation comparison without private recordings or provider access.
