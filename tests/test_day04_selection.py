"""Selection follows the fixed Day04 candidate order on exact CV ties."""

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.models.run_experiments_day04 import TIE_ORDER, select_cv_winner


def test_exact_tie_keeps_baseline() -> None:
    summary = pd.DataFrame({
        "candidate_id": TIE_ORDER,
        "mean_macro_f1": [0.7] * len(TIE_ORDER),
    })

    winner, tied = select_cv_winner(summary)

    assert winner == "baseline"
    assert tied == TIE_ORDER


def test_candidate_tie_uses_spec_order_after_baseline() -> None:
    scores = {candidate: 0.5 for candidate in TIE_ORDER}
    scores["baseline"] = 0.6
    scores["linear_svc"] = 0.8
    scores["word_unigram"] = 0.8
    summary = pd.DataFrame({"candidate_id": list(scores), "mean_macro_f1": list(scores.values())})

    winner, tied = select_cv_winner(summary)

    assert winner == "word_unigram"
    assert tied == ("word_unigram", "linear_svc")


@pytest.mark.parametrize(
    "summary",
    [
        pd.DataFrame({"candidate_id": ["baseline"], "mean_macro_f1": [0.4]}),
        pd.DataFrame({"candidate_id": TIE_ORDER, "mean_macro_f1": [0.4, 0.5, 0.6, 0.7, 0.8, float("nan")]}),
    ],
)
def test_selection_rejects_incomplete_or_nonfinite_scores(summary: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="finite mean macro F1"):
        select_cv_winner(summary)
