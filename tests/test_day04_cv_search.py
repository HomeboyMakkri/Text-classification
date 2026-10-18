"""Tests for the bounded, fixed-fold LogisticRegression C search."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.data.baseline_config import load_baseline_config
from finnews_sentiment.data.experiment_splits import CVFoldSet, build_train_cv_folds
from finnews_sentiment.models.cv_runner import run_cv_candidate
from finnews_sentiment.models.cv_search import search_logreg_c
from finnews_sentiment.models.train_model import build_baseline_pipeline


@pytest.fixture
def train_and_folds() -> tuple[pd.DataFrame, CVFoldSet]:
    class_names = ("negative", "neutral", "positive")
    rows: list[tuple[int, str, str, str, int]] = []
    for source_id in range(90):
        label_id = source_id % len(class_names)
        sentiment = class_names[label_id]
        text = f"company item{source_id} market {sentiment} signal"
        rows.append((source_id, text, sentiment, text, label_id))
    train = pd.DataFrame(
        rows,
        columns=["source_row_id", "text", "sentiment", "text_clean", "label_id"],
    )
    manifest = pd.DataFrame({
        "source_row_id": train["source_row_id"],
        "group_id": train["source_row_id"],
        "split": "train",
    })
    return train, build_train_cv_folds(train, manifest)


def test_search_uses_fixed_c_grid_and_c_one_matches_cv_runner(
    train_and_folds: tuple[pd.DataFrame, CVFoldSet],
) -> None:
    train, folds = train_and_folds
    config = load_baseline_config(ROOT / "configs/train_baseline.json")
    pipeline = build_baseline_pipeline(config)

    search_result = search_logreg_c(pipeline, train, folds)
    baseline_result = run_cv_candidate("baseline-control", pipeline, train, folds)

    assert search_result.scores["C"].tolist() == [0.1, 1.0, 10.0]
    assert search_result.n_splits == folds.n_splits
    assert search_result.random_state == folds.random_state
    assert not search_result.warnings
    assert not hasattr(pipeline.named_steps["tfidf"], "vocabulary_")
    c_one = search_result.scores.loc[search_result.scores["C"].eq(1.0)].iloc[0]
    observed = np.asarray([c_one[f"fold_{fold_id}_macro_f1"] for fold_id in range(folds.n_splits)])
    np.testing.assert_allclose(observed, baseline_result.fold_scores["macro_f1"].to_numpy())
    observed_accuracy = np.asarray([
        c_one[f"fold_{fold_id}_accuracy"] for fold_id in range(folds.n_splits)
    ])
    np.testing.assert_allclose(observed_accuracy, baseline_result.fold_scores["accuracy"].to_numpy())
    assert c_one["mean_macro_f1"] == pytest.approx(baseline_result.mean_macro_f1)
    assert c_one["std_macro_f1"] == pytest.approx(baseline_result.std_macro_f1)


def test_search_scores_are_repeatable(train_and_folds: tuple[pd.DataFrame, CVFoldSet]) -> None:
    train, folds = train_and_folds
    pipeline = build_baseline_pipeline(load_baseline_config(ROOT / "configs/train_baseline.json"))

    first = search_logreg_c(pipeline, train, folds)
    second = search_logreg_c(pipeline, train, folds)

    pd.testing.assert_frame_equal(
        first.scores.drop(columns="mean_fit_seconds"),
        second.scores.drop(columns="mean_fit_seconds"),
    )
