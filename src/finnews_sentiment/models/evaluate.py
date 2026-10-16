"""Validation metrics for aligned three-class baseline predictions."""

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from finnews_sentiment.data.load_data import CLASS_ORDER

CLASS_IDS = (0, 1, 2)
# scikit-learn 1.9 accepts integer 0 here at runtime, but its current stubs only
# declare str. Keep the runtime contract while narrowing the typing workaround.
_ZERO_DIVISION: str = cast(str, 0)


@dataclass(frozen=True)
class ClassMetrics:
    """Metrics and support for one class in the fixed sentiment order."""

    class_id: int
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationMetrics:
    """Aggregate and per-class metrics for one nonempty evaluation split."""

    n_samples: int
    accuracy: float
    macro_f1: float
    per_class: tuple[ClassMetrics, ...]


@dataclass(frozen=True)
class BaselineEvaluation:
    """Metrics for both baselines and their fixed-column comparison table."""

    baseline: ClassificationMetrics
    dummy: ClassificationMetrics
    comparison: pd.DataFrame


def _label_ids(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    """Validate one-dimensional integer class IDs without coercing labels."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional sequence of class IDs") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    ids: list[int] = []
    for value in array:
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} values must be integer class IDs, got {value!r}")
        label_id = int(value)
        if label_id not in CLASS_IDS:
            raise ValueError(f"{name} contains unknown class ID {label_id}; expected {CLASS_IDS}")
        ids.append(label_id)
    return np.asarray(ids, dtype=np.int64)


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> ClassificationMetrics:
    """Compute accuracy, macro F1 and per-class metrics in fixed class order."""
    truth = _label_ids(y_true, "y_true")
    predictions = _label_ids(y_pred, "y_pred")
    if len(truth) != len(predictions):
        raise ValueError(f"y_true and y_pred lengths differ: {len(truth)} != {len(predictions)}")

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=list(CLASS_IDS), average=None, zero_division=_ZERO_DIVISION,
    )
    expected_shape = (len(CLASS_IDS),)
    if not isinstance(precision, np.ndarray) or precision.shape != expected_shape:
        raise RuntimeError("sklearn returned unexpected per-class precision shape")
    if not isinstance(recall, np.ndarray) or recall.shape != expected_shape:
        raise RuntimeError("sklearn returned unexpected per-class recall shape")
    if not isinstance(f1, np.ndarray) or f1.shape != expected_shape:
        raise RuntimeError("sklearn returned unexpected per-class F1 shape")
    if support is None or support.shape != expected_shape:
        raise RuntimeError("sklearn returned unexpected per-class metric shapes")
    per_class = tuple(
        ClassMetrics(
            class_id=class_id,
            class_name=class_name,
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, (class_id, class_name) in enumerate(zip(CLASS_IDS, CLASS_ORDER, strict=True))
    )
    return ClassificationMetrics(
        n_samples=len(truth),
        accuracy=float(accuracy_score(truth, predictions)),
        macro_f1=float(f1_score(
            truth, predictions, labels=list(CLASS_IDS), average="macro", zero_division=_ZERO_DIVISION,
        )),
        per_class=per_class,
    )


def evaluate_baselines(
    y_true: Sequence[int] | np.ndarray,
    baseline_predictions: Sequence[int] | np.ndarray,
    dummy_predictions: Sequence[int] | np.ndarray,
) -> BaselineEvaluation:
    """Score both models on the same validation truth and build a comparison table."""
    truth = _label_ids(y_true, "y_true")
    baseline = _label_ids(baseline_predictions, "baseline_predictions")
    dummy = _label_ids(dummy_predictions, "dummy_predictions")
    if len(truth) != len(baseline) or len(truth) != len(dummy):
        raise ValueError(
            "y_true, baseline_predictions and dummy_predictions must have equal lengths: "
            f"{len(truth)}, {len(baseline)}, {len(dummy)}"
        )
    baseline_metrics = evaluate_predictions(truth, baseline)
    dummy_metrics = evaluate_predictions(truth, dummy)
    rows: list[dict[str, int | float | str]] = []
    for name, metrics in (("baseline", baseline_metrics), ("dummy", dummy_metrics)):
        row: dict[str, int | float | str] = {
            "model": name,
            "n_samples": metrics.n_samples,
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
        }
        for class_metrics in metrics.per_class:
            for metric_name in ("precision", "recall", "f1", "support"):
                value = getattr(class_metrics, metric_name)
                row[f"{class_metrics.class_name}_{metric_name}"] = value
        rows.append(row)
    return BaselineEvaluation(
        baseline=baseline_metrics,
        dummy=dummy_metrics,
        comparison=pd.DataFrame(rows),
    )
