"""Runpod queue-based serverless entry point for the Echora Whisper worker.

Runpod's GitHub deploy check looks for `runpod.serverless.start()` in a
root-level module, so the entry point lives here rather than beside the worker
implementation. The implementation stays in `backend/runpod_worker/`, and the
container image mirrors this repository's layout, so the import below resolves
identically in a local checkout and inside the built image.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The worker package lives under backend/, which is not importable by default
# from this file's directory, so the path bootstrap must precede the import.
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from runpod_worker.worker import handler, runtime


def main() -> None:
    """Preload the model, then hand requests to Runpod's queue-based runtime."""

    import runpod

    runtime.load()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
