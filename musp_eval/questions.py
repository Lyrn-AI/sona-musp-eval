"""The 238 MuSP-Bench questions that can be answered from audio alone, and the two halves we report.

MuSP-Bench tags every question with a modality: S (score only), P (performance only), S&P (both jointly) and
S/P (either). S and S&P need the score, so an audio-only evaluation keeps P (106) and S/P (132, in three
submodalities: general 36, tonality/style 48, composer/title 48).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SUBSETS = ["P", "general", "tonality/style", "composer/title"]


def load_questions(musp_dir: str | Path = "MuSP-Bench") -> pd.DataFrame:
    """The 238 audio-only questions with a `subset` column (P or the S/P submodality)."""
    q = pd.read_csv(Path(musp_dir) / "data/questions.csv")
    q = q[q.modality.isin(["P", "S/P"])].copy()
    q["subset"] = q.modality.where(q.modality == "P", q.submodality)
    assert len(q) == 238 and set(q.subset) == set(SUBSETS)
    return q


def half(piece_number: int) -> str:
    """Design choices were made on the odd-numbered pieces only; the even-numbered pieces are held out."""
    return "development" if piece_number % 2 == 1 else "held-out"


def question_block(row) -> str:
    """The question text every system receives, after its system prompt and inputs."""
    return (
        f"Question: {row['question']}\n\n"
        f"Answer format: {row['answer_format']}\n"
        f"Answer quantity: {row['answer_quantity']}\n"
        f"Syntax example (shows the format only; it is not the answer): {row['answer_example']}"
    )
