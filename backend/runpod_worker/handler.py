"""Runpod queue-based serverless process entry point.

This file intentionally sits beside the Dockerfile selected in Runpod's GitHub
deployment flow. Keeping the startup call in that directory lets Runpod's
repository validation find it, while ``worker.py`` remains importable and
unit-testable as the inference implementation.
"""

from __future__ import annotations

from worker import handler, runtime


def main() -> None:
    """Preload the model once, then start the Runpod worker loop."""

    import runpod

    runtime.load()
    print(
        f"Loaded model {runtime.config.model_id} from cached revision {runtime.model_revision}.",
        flush=True,
    )
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
