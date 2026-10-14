"""Builders and train-only fit API for Day03 baseline estimators."""

from dataclasses import dataclass
from itertools import combinations
import warnings
from typing import cast

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from finnews_sentiment.data.baseline_config import BaselineConfig
from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.data.load_processed import validate_processed_news
from finnews_sentiment.data.split_data import NewsSplits, SPLIT_ORDER


@dataclass(frozen=True)
class BaselineFit:
    """Fitted estimators and predictions aligned to validation source IDs."""

    pipeline: Pipeline
    dummy: DummyClassifier
    validation_predictions: pd.DataFrame
    train_feature_shape: tuple[int, int]
    feature_count: int
    convergence_status: str


def build_baseline_pipeline(config: BaselineConfig) -> Pipeline:
    """Return a new, unfitted TF-IDF and LogisticRegression pipeline."""
    tfidf = TfidfVectorizer(**config.tfidf)
    classifier = LogisticRegression(**config.logreg)
    return Pipeline([("tfidf", tfidf), ("clf", classifier)])


def build_dummy_classifier(config: BaselineConfig) -> DummyClassifier:
    """Return a new, unfitted majority-class DummyClassifier."""
    return DummyClassifier(**config.dummy)


def _validate_splits(splits: NewsSplits) -> None:
    """Validate prepared splits before constructing or fitting estimators."""
    if set(splits.splits) != set(SPLIT_ORDER):
        raise ValueError(f"splits must have exactly these keys: {SPLIT_ORDER}")
    manifest = splits.manifest
    manifest_columns = ("source_row_id", "group_id", "split")
    if tuple(manifest.columns) != manifest_columns or manifest.empty:
        raise ValueError(f"manifest must be nonempty with columns {manifest_columns}")
    if manifest["source_row_id"].duplicated().any():
        raise ValueError("manifest source_row_id values must be unique")
    if not manifest["split"].isin(SPLIT_ORDER).all():
        raise ValueError("manifest contains an unknown split name")

    manifest_ids = manifest["source_row_id"].tolist()
    observed_ids: list[int] = []
    all_rows: list[pd.DataFrame] = []
    group_to_split: dict[int, str] = {}
    for split_name in SPLIT_ORDER:
        frame = splits.splits[split_name]
        validate_processed_news(frame)
        ids = frame["source_row_id"].tolist()
        expected_ids = manifest.loc[manifest["split"] == split_name, "source_row_id"].tolist()
        if ids != expected_ids:
            raise ValueError(f"{split_name} rows do not match manifest IDs/order")
        if set(frame["sentiment"]) != set(CLASS_ORDER):
            raise ValueError(f"{split_name} must contain all three classes")
        row_groups = manifest.loc[manifest["split"] == split_name, "group_id"].tolist()
        if len(row_groups) != len(frame):
            raise ValueError(f"{split_name} manifest group IDs do not align with rows")
        for group_id in row_groups:
            if isinstance(group_id, bool) or not isinstance(group_id, (int, np.integer)):
                raise ValueError("manifest group_id values must be integers")
            previous = group_to_split.setdefault(int(group_id), split_name)
            if previous != split_name:
                raise ValueError("A text group crosses split boundaries")
        observed_ids.extend(ids)
        all_rows.append(frame)

    if set(observed_ids) != set(manifest_ids) or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("splits must cover each manifest source ID exactly once")
    for left, right in combinations(SPLIT_ORDER, 2):
        left_frame, right_frame = splits.splits[left], splits.splits[right]
        for column in ("source_row_id", "text", "text_clean"):
            if not set(left_frame[column]).isdisjoint(right_frame[column]):
                raise ValueError(f"{column} values must be disjoint across splits")

    # Confirm group IDs identify the minimum source_row_id for each exact clean text.
    source_to_group = dict(zip(manifest["source_row_id"], manifest["group_id"], strict=True))
    clean_group_ids: dict[str, set[int]] = {}
    for frame, split_name in zip(all_rows, SPLIT_ORDER, strict=True):
        entries = manifest.loc[manifest["split"] == split_name]
        clean_to_source_ids: dict[str, list[int]] = {}
        for clean, group_id in zip(frame["text_clean"], entries["group_id"], strict=True):
            clean_group_ids.setdefault(clean, set()).add(int(group_id))
        for clean, source_id in zip(frame["text_clean"], frame["source_row_id"], strict=True):
            clean_to_source_ids.setdefault(clean, []).append(int(source_id))
        for clean, source_ids in clean_to_source_ids.items():
            expected_group = min(source_ids)
            if any(int(source_to_group[source_id]) != expected_group for source_id in source_ids):
                raise ValueError("manifest group IDs must equal the minimum source_row_id per clean text")
    if any(len(group_ids) != 1 for group_ids in clean_group_ids.values()):
        raise ValueError("Identical clean texts must map to one manifest group ID")


def fit_baselines(splits: NewsSplits, config: BaselineConfig) -> BaselineFit:
    """Fit baseline models on train only and predict the aligned validation rows.

    The held-out test split is structurally checked, but its texts and labels
    are never passed to a transformer or estimator.
    """
    _validate_splits(splits)
    train = splits.splits["train"]
    validation = splits.splits["validation"]
    train_texts = train[config.text_col].tolist()
    train_labels = train[config.target_col].to_numpy(dtype=np.int64)
    validation_texts = validation[config.text_col].tolist()

    pipeline = build_baseline_pipeline(config)
    dummy = build_dummy_classifier(config)
    train_constant_features = np.ones((len(train), 1), dtype=np.float64)
    validation_constant_features = np.ones((len(validation), 1), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            pipeline.fit(train_texts, train_labels)
        except ConvergenceWarning as error:
            raise RuntimeError(
                "LogisticRegression did not converge with the fixed baseline config; D3-04 is unconfirmed"
            ) from error
        dummy.fit(train_constant_features, train_labels)

    vectorizer = pipeline.named_steps["tfidf"]
    train_features = vectorizer.transform(train_texts)
    if not issparse(train_features):
        raise RuntimeError("TF-IDF output must remain sparse")
    # Single-column label targets guarantee one-dimensional ndarray outputs;
    # sklearn's stubs also describe multi-output list results.
    pipeline_predictions = cast(np.ndarray, pipeline.predict(validation_texts))
    dummy_predictions = cast(np.ndarray, dummy.predict(validation_constant_features))
    expected_classes = np.asarray([0, 1, 2], dtype=np.int64)
    if not np.array_equal(pipeline.named_steps["clf"].classes_, expected_classes):
        raise RuntimeError("LogisticRegression classes_ do not match fixed class IDs [0, 1, 2]")
    dummy_classes = cast(np.ndarray, dummy.classes_)
    if not np.array_equal(dummy_classes, expected_classes):
        raise RuntimeError("DummyClassifier classes_ do not match fixed class IDs [0, 1, 2]")
    valid_predictions = set(expected_classes.tolist())
    if not set(pipeline_predictions.tolist()).issubset(valid_predictions) or not set(dummy_predictions.tolist()).issubset(valid_predictions):
        raise RuntimeError("Estimator produced a prediction outside the fixed class IDs [0, 1, 2]")
    prediction_frame = pd.DataFrame({
        "source_row_id": validation["source_row_id"].to_numpy(copy=True),
        "baseline_prediction": pipeline_predictions,
        "dummy_prediction": dummy_predictions,
    })
    return BaselineFit(
        pipeline=pipeline,
        dummy=dummy,
        validation_predictions=prediction_frame,
        train_feature_shape=(int(train_features.shape[0]), int(train_features.shape[1])),
        feature_count=int(train_features.shape[1]),
        convergence_status="converged_no_warning",
    )
