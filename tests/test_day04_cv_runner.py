"""Leakage-safe one-candidate CV runner tests."""

from dataclasses import replace
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import finnews_sentiment.models.cv_runner as cv_runner
from finnews_sentiment.data.baseline_config import load_baseline_config
from finnews_sentiment.data.experiment_splits import CVFoldSet, build_train_cv_folds
from finnews_sentiment.models.cv_runner import run_cv_candidate
from finnews_sentiment.models.train_model import build_baseline_pipeline


class WarningClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, features: object, labels: object) -> WarningClassifier:
        warnings.warn("test convergence warning", ConvergenceWarning, stacklevel=2)
        return self

    def predict(self, features: object) -> np.ndarray:
        return np.zeros(1, dtype=np.int64)


@pytest.fixture
def cv_data() -> tuple[pd.DataFrame, CVFoldSet]:
    class_order = ("negative", "neutral", "positive")
    rows: list[tuple[int, str, str, str, int]] = []
    for source_id in range(60):
        label_id = source_id % 3
        text = f"company item{source_id} outlook {class_order[label_id]}"
        clean = text if source_id != 0 else f"{text} exclusivefoldword"
        rows.append((source_id, text, class_order[label_id], clean, label_id))
    train = pd.DataFrame(
        rows, columns=["source_row_id", "text", "sentiment", "text_clean", "label_id"],
    )
    train_manifest = pd.DataFrame({
        "source_row_id": train["source_row_id"],
        "group_id": train["source_row_id"],
        "split": "train",
    })
    return train, build_train_cv_folds(train, train_manifest)


def test_runner_clones_pipeline_and_scores_fixed_folds(cv_data: tuple[pd.DataFrame, CVFoldSet]) -> None:
    train, folds = cv_data
    config = load_baseline_config(ROOT / "configs/train_baseline.json")
    pipeline = build_baseline_pipeline(config)
    result = run_cv_candidate("baseline", pipeline, train, folds)

    assert result.candidate_id == "baseline"
    assert result.n_splits == 3 and result.random_state == 42
    assert result.class_order == ("negative", "neutral", "positive")
    assert result.fold_scores.index.tolist() == [0, 1, 2]
    assert result.fold_scores["n_features"].gt(0).all()
    assert result.fold_scores["macro_f1"].between(0, 1).all()
    assert result.fold_scores["accuracy"].between(0, 1).all()
    assert result.fold_scores["fit_seconds"].ge(0).all()
    assert result.fold_scores["score_seconds"].ge(0).all()
    assert result.fold_scores["convergence_status"].eq("converged_no_warning").all()
    assert result.mean_macro_f1 == pytest.approx(result.fold_scores["macro_f1"].mean())
    assert result.std_macro_f1 == pytest.approx(result.fold_scores["macro_f1"].std(ddof=0))
    assert not hasattr(pipeline.named_steps["tfidf"], "vocabulary_")
    assert not hasattr(pipeline.named_steps["clf"], "classes_")


def test_validation_only_term_never_enters_heldout_fold_vocabulary(
    cv_data: tuple[pd.DataFrame, CVFoldSet], monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, folds = cv_data
    config = load_baseline_config(ROOT / "configs/train_baseline.json")
    pipeline = build_baseline_pipeline(config)
    original_clone = cv_runner.clone
    clones: list[Pipeline] = []

    def record_clone(estimator: Pipeline) -> Pipeline:
        fitted_candidate = original_clone(estimator)
        clones.append(fitted_candidate)
        return fitted_candidate

    monkeypatch.setattr(cv_runner, "clone", record_clone)
    result = run_cv_candidate("baseline", pipeline, train, folds)

    assert len(clones) == 3 and len({id(candidate) for candidate in clones}) == 3
    source_id = 0
    source_position = train["source_row_id"].tolist().index(source_id)
    heldout_fold = next(
        fold.fold_id for fold in folds.folds if source_position in fold.validation_positions
    )
    heldout_vectorizer = clones[heldout_fold].named_steps["tfidf"]
    assert isinstance(heldout_vectorizer, TfidfVectorizer)
    assert "exclusivefoldword" not in heldout_vectorizer.vocabulary_
    assert result.fold_scores.loc[heldout_fold, "n_features"] == len(heldout_vectorizer.vocabulary_)


def test_repeated_runs_have_identical_metric_scores(cv_data: tuple[pd.DataFrame, CVFoldSet]) -> None:
    train, folds = cv_data
    pipeline = build_baseline_pipeline(load_baseline_config(ROOT / "configs/train_baseline.json"))
    first = run_cv_candidate("baseline", pipeline, train, folds)
    second = run_cv_candidate("baseline", pipeline, train, folds)

    pd.testing.assert_series_equal(first.fold_scores["macro_f1"], second.fold_scores["macro_f1"])
    pd.testing.assert_series_equal(first.fold_scores["accuracy"], second.fold_scores["accuracy"])


def test_malformed_fold_is_rejected_before_cloning(
    cv_data: tuple[pd.DataFrame, CVFoldSet], monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, folds = cv_data
    first = folds.folds[0]
    malformed = replace(first, validation_positions=first.validation_positions[:-1])
    bad_folds = replace(folds, folds=(malformed, *folds.folds[1:]))
    pipeline = build_baseline_pipeline(load_baseline_config(ROOT / "configs/train_baseline.json"))

    def should_not_clone(_: Pipeline) -> Pipeline:
        raise AssertionError("Malformed folds must fail before candidate cloning")

    monkeypatch.setattr(cv_runner, "clone", should_not_clone)
    with pytest.raises(ValueError, match="partition every train row exactly once"):
        run_cv_candidate("baseline", pipeline, train, bad_folds)


def test_prefitted_pipeline_is_rejected(cv_data: tuple[pd.DataFrame, CVFoldSet]) -> None:
    train, folds = cv_data
    pipeline = build_baseline_pipeline(load_baseline_config(ROOT / "configs/train_baseline.json"))
    pipeline.fit(train["text_clean"].tolist(), train["label_id"].to_numpy(dtype=np.int64))

    with pytest.raises(ValueError, match="must be unfitted"):
        run_cv_candidate("baseline", pipeline, train, folds)


def test_convergence_warning_fails_with_candidate_and_fold_context(
    cv_data: tuple[pd.DataFrame, CVFoldSet],
) -> None:
    train, folds = cv_data
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", WarningClassifier())])

    with pytest.raises(RuntimeError, match="candidate 'warning-model' fold 0 raised ConvergenceWarning"):
        run_cv_candidate("warning-model", pipeline, train, folds)
