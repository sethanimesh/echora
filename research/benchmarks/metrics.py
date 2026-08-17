"""Literal ASR metrics for Echora benchmark result files.

This module deliberately scores recognizer output before any semantic or
personal-context component sees it.  A result file may contain raw acoustic
alternatives, but the normal WER/CER columns always describe rank 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar


BENCH_DIR = Path(__file__).resolve().parent
WORD_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
T = TypeVar("T")


def normalize(text: str) -> str:
    """Apply one transparent, punctuation-insensitive ASR normalization."""
    text = unicodedata.normalize("NFKC", text).lower().replace("\u2019", "'")
    return " ".join(WORD_RE.findall(text))


def edit_counts(reference: Sequence[T], hypothesis: Sequence[T]) -> tuple[int, int, int]:
    """Return substitutions, deletions, insertions for a minimum edit path."""
    # Each cell stores (total edits, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref_item in enumerate(reference, start=1):
        current = [(i, 0, i, 0)]
        for j, hyp_item in enumerate(hypothesis, start=1):
            if ref_item == hyp_item:
                current.append(previous[j - 1])
                continue
            sub = previous[j - 1]
            delete = previous[j]
            insert = current[j - 1]
            choices = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            # Stable tie breaking makes detailed error counts reproducible.
            current.append(min(choices, key=lambda item: (item[0], item[2] + item[3], item[1])))
        previous = current
    return previous[-1][1:]


@dataclass(frozen=True)
class UtteranceScore:
    word_substitutions: int
    word_deletions: int
    word_insertions: int
    reference_words: int
    character_errors: int
    reference_characters: int

    @property
    def word_errors(self) -> int:
        return self.word_substitutions + self.word_deletions + self.word_insertions

    @property
    def wer(self) -> float:
        return self.word_errors / self.reference_words if self.reference_words else 0.0

    @property
    def cer(self) -> float:
        return self.character_errors / self.reference_characters if self.reference_characters else 0.0


def score(reference: str, hypothesis: str) -> UtteranceScore:
    ref_normalized = normalize(reference)
    hyp_normalized = normalize(hypothesis)
    ref_words = ref_normalized.split()
    hyp_words = hyp_normalized.split()
    substitutions, deletions, insertions = edit_counts(ref_words, hyp_words)
    char_counts = edit_counts(list(ref_normalized), list(hyp_normalized))
    return UtteranceScore(
        word_substitutions=substitutions,
        word_deletions=deletions,
        word_insertions=insertions,
        reference_words=len(ref_words),
        character_errors=sum(char_counts),
        reference_characters=len(ref_normalized),
    )


def alternatives_from_row(row: dict[str, str]) -> list[str]:
    hypotheses = [row.get("transcript", "")]
    encoded = row.get("alternatives", "").strip()
    if encoded:
        parsed = json.loads(encoded)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("alternatives must be a JSON list of strings")
        hypotheses.extend(parsed)
    return hypotheses


def aggregate(scores: Sequence[UtteranceScore]) -> UtteranceScore:
    return UtteranceScore(
        word_substitutions=sum(item.word_substitutions for item in scores),
        word_deletions=sum(item.word_deletions for item in scores),
        word_insertions=sum(item.word_insertions for item in scores),
        reference_words=sum(item.reference_words for item in scores),
        character_errors=sum(item.character_errors for item in scores),
        reference_characters=sum(item.reference_characters for item in scores),
    )


@dataclass(frozen=True)
class FileScore:
    path: Path
    utterances: int
    top1: UtteranceScore
    oracle: UtteranceScore


def score_file(path: Path) -> FileScore:
    top1_scores: list[UtteranceScore] = []
    oracle_scores: list[UtteranceScore] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            reference = row.get("reference", "").strip()
            if not reference:
                continue
            candidates = alternatives_from_row(row)
            candidate_scores = [score(reference, item) for item in candidates]
            top1_scores.append(candidate_scores[0])
            oracle_scores.append(min(candidate_scores, key=lambda item: (item.word_errors, item.character_errors)))
    return FileScore(path, len(top1_scores), aggregate(top1_scores), aggregate(oracle_scores))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score literal ASR TSVs before any candidate-ranking layer."
    )
    parser.add_argument("results", nargs="*", type=Path, help="TSV files (default: results/*.tsv)")
    args = parser.parse_args()
    paths = args.results or sorted((BENCH_DIR / "results").glob("*.tsv"))
    if not paths:
        parser.error("no result TSV files found")

    print("model\tutterances\tWER\tCER\tS\tD\tI\ttop-k oracle WER")
    for path in paths:
        result = score_file(path)
        top1 = result.top1
        print(
            f"{path.stem}\t{result.utterances}\t{top1.wer:.1%}\t{top1.cer:.1%}\t"
            f"{top1.word_substitutions}\t{top1.word_deletions}\t{top1.word_insertions}\t"
            f"{result.oracle.wer:.1%}"
        )


if __name__ == "__main__":
    main()
