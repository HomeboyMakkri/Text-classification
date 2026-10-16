"""Run one unfitted text-classification Pipeline on fixed CV folds."""

from dataclasses import dataclass
import time
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from finnews_sentiment.data.experiment_splits import CVFold, CVFoldSet, CV_MANIFEST_COLUMNS
from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.data.load_processed import validate_processed_news
from finnews_sentiment.models.evaluate import evaluate_predictions


@dataclass(frozen=True)
class CVRunResult:
    """Fold-level validation metrics and run context for one candidate."""

    candidate_id: str
    parameters: dict[str, object]
    fold_scores: pd.DataFrame
    mean_macro_f1: float
    std_macro_f1: float
    mean_accuracy: float
    std_accuracy: float
    n_splits: int
    random_state: int
    class_order: tuple[str, ...]


def _validate_folds(train: pd.DataFrame, fold_set: CVFoldSet) -> None:
    """Reject malformed folds before cloning or fitting any candidate."""
    if tuple(train.columns) != ("source_row_id", "text", "sentiment", "text_clean", "label_id"):
        raise ValueError("CV train must have the exact processed Day03 schema")
    validate_processed_news(train)
    if tuple(fold_set.manifest.columns) != CV_MANIFEST_COLUMNS:
        raise ValueError(f"CV fold manifest columns must be {CV_MANIFEST_COLUMNS}")
    if fold_set.n_splits < 2 or len(fold_set.folds) != fold_set.n_splits:
        raise ValueError("CV fold count does not match n_splits")
    if fold_set.random_state < 0 or fold_set.random_state >= 2**32:
        raise ValueError("CV random_state must be in [0, 2**32 - 1]")
    if fold_set.manifest["source_row_id"].tolist() != train["source_row_id"].tolist():
        raise ValueError("CV fold manifest source IDs/order do not match train")
    clean_to_group = train.groupby("text_clean", sort=False)["source_row_id"].min().astype("int64").to_dict()
    expected_groups = train["text_clean"].map(clean_to_group).tolist()
    if fold_set.manifest["group_id"].tolist() != expected_groups:
        raise ValueError("CV fold manifest groups do not match exact-text train groups")
    if not fold_set.manifest["fold_id"].isin(range(fold_set.n_splits)).all():
        raise ValueError("CV fold manifest contains an unknown fold_id")
    expected_positions = set(range(len(train)))
    labels = train["label_id"].to_numpy(dtype=np.int64)
    groups = fold_set.manifest["group_id"].to_numpy(dtype=np.int64)
    validation_assignments = [-1] * len(train)
    for expected_fold_id, fold in enumerate(fold_set.folds):
        if fold.fold_id != expected_fold_id:
            raise ValueError("CV fold IDs must be contiguous and ordered from zero")
        train_positions = list(fold.train_positions)
        validation_positions = list(fold.validation_positions)
        if (
            len(set(train_positions)) != len(train_positions)
            or len(set(validation_positions)) != len(validation_positions)
            or set(train_positions).intersection(validation_positions)
            or set(train_positions).union(validation_positions) != expected_positions
        ):
            raise ValueError(f"CV fold {fold.fold_id} positions must partition every train row exactly once")
        if any(position < 0 or position >= len(train) for position in (*train_positions, *validation_positions)):
            raise ValueError(f"CV fold {fold.fold_id} contains an out-of-range row position")
        if set(groups[train_positions]).intersection(groups[validation_positions]):
            raise ValueError(f"CV fold {fold.fold_id} leaks text groups across train/validation")
        if set(labels[train_positions].tolist()) != {0, 1, 2} or set(labels[validation_positions].tolist()) != {0, 1, 2}:
            raise ValueError(f"CV fold {fold.fold_id} must contain all three classes in both partitions")
        expected_validation = fold_set.manifest["fold_id"].eq(fold.fold_id).to_numpy().nonzero()[0].tolist()
        if validation_positions != expected_validation:
            raise ValueError(f"CV fold {fold.fold_id} positions differ from the fold manifest")
        for position in validation_positions:
            if validation_assignments[position] != -1:
                raise ValueError("CV validation folds overlap in train rows")
            validation_assignments[position] = fold.fold_id
    if any(fold_id < 0 for fold_id in validation_assignments):
        raise ValueError("CV validation folds do not cover every Day03 train row exactly once")


def run_cv_candidate(
    candidate_id: str,
    pipeline: Pipeline,
    train: pd.DataFrame,
    fold_set: CVFoldSet,
) -> CVRunResult:
    """Fit one fresh clone per fold and score that fold's rows only.

    The API accepts Day03 train and fixed folds only. It has no parameters for
    outer validation/test data. Fit/score failures and convergence warnings
    raise with candidate/fold context rather than returning partial or NaN scores.
    """
    if not candidate_id.strip():
        raise ValueError("candidate_id must be a nonempty string")
    if not isinstance(pipeline, Pipeline):
        raise TypeError("CV candidates must be sklearn Pipeline instances")
    try:
        check_is_fitted(pipeline)
    except NotFittedError:
        pass
    else:
        raise ValueError("CV candidate Pipeline must be unfitted")
    _validate_folds(train, fold_set)

    score_rows: list[dict[str, object]] = []
    for fold in fold_set.folds:
        fold_pipeline = clone(pipeline)
        train_part = train.iloc[list(fold.train_positions)]
        validation_part = train.iloc[list(fold.validation_positions)]
        train_texts = train_part["text_clean"].tolist()
        validation_texts = validation_part["text_clean"].tolist()
        train_labels = train_part["label_id"].to_numpy(dtype=np.int64)
        validation_labels = validation_part["label_id"].to_numpy(dtype=np.int64)
        captured_warnings: list[str] = []
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                fold_pipeline.fit(train_texts, train_labels)
            except ConvergenceWarning as error:
                raise RuntimeError(
                    f"CV candidate {candidate_id!r} fold {fold.fold_id} raised ConvergenceWarning"
                ) from error
            except Exception as error:
                raise RuntimeError(
                    f"CV candidate {candidate_id!r} fold {fold.fold_id} fit failed: {error}"
                ) from error
            fit_seconds = time.perf_counter() - fit_started
            score_started = time.perf_counter()
            try:
                vectorizer = fold_pipeline.named_steps.get("tfidf")
                if not isinstance(vectorizer, TfidfVectorizer):
                    raise ValueError("Candidate Pipeline must contain a TfidfVectorizer step named 'tfidf'")
                train_features = vectorizer.transform(train_texts)
                validation_features = vectorizer.transform(validation_texts)
                if not issparse(train_features) or not issparse(validation_features):
                    raise RuntimeError("TF-IDF feature matrices must remain sparse")
                predictions = fold_pipeline.predict(validation_texts)
                metrics = evaluate_predictions(validation_labels, predictions)
            except ConvergenceWarning as error:
                raise RuntimeError(
                    f"CV candidate {candidate_id!r} fold {fold.fold_id} raised ConvergenceWarning while scoring"
                ) from error
            except Exception as error:
                raise RuntimeError(
                    f"CV candidate {candidate_id!r} fold {fold.fold_id} scoring failed: {error}"
                ) from error
            score_seconds = time.perf_counter() - score_started
        captured_warnings.extend(
            f"{item.category.__name__}: {item.message}" for item in recorded_warnings
        )
        score_rows.append({
            "fold_id": fold.fold_id,
            "train_rows": len(train_part),
            "train_groups": int(fold_set.manifest.iloc[list(fold.train_positions)]["group_id"].nunique()),
            "validation_rows": len(validation_part),
            "validation_groups": int(fold_set.manifest.iloc[list(fold.validation_positions)]["group_id"].nunique()),
            "n_features": len(vectorizer.vocabulary_),
            "macro_f1": metrics.macro_f1,
            "accuracy": metrics.accuracy,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "convergence_status": "converged_no_warning",
            "warnings": tuple(captured_warnings),
        })

    fold_scores = pd.DataFrame(score_rows).set_index("fold_id")
    macro_scores = fold_scores["macro_f1"].to_numpy(dtype=np.float64)
    accuracy_scores = fold_scores["accuracy"].to_numpy(dtype=np.float64)
    return CVRunResult(
        candidate_id=candidate_id,
        parameters=pipeline.get_params(deep=True),
        fold_scores=fold_scores,
        mean_macro_f1=float(np.mean(macro_scores)),
        std_macro_f1=float(np.std(macro_scores, ddof=0)),
        mean_accuracy=float(np.mean(accuracy_scores)),
        std_accuracy=float(np.std(accuracy_scores, ddof=0)),
        n_splits=fold_set.n_splits,
        random_state=fold_set.random_state,
        class_order=CLASS_ORDER,
    )
