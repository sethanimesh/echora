"""Reusable facade over the same language pipeline used by the application.

Provider composition belongs to this library's explicit Config. The application
keeps its async, fixed-provider, revision-bound adapter independently.
"""
from __future__ import annotations

from communication.backend.hinglish_core.pipeline import Pipeline as CorePipeline, Result
from communication.backend.hinglish_core.classify.lexical import LexicalClassifier
from echora.config import Config


class Pipeline(CorePipeline):
    @classmethod
    def from_config(cls, config: Config | None = None) -> Pipeline:
        cfg = config or Config.from_env()
        return cls(
            lexical=LexicalClassifier(protected_terms=cfg.protected_terms),
            classifier=cfg.build_classifier(),
        )


__all__ = ["Pipeline", "Result"]
