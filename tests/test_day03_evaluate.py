"""Metric formulas and input contracts for D3-05."""

from pathlib import Path
import sys
from typing import Sequence, cast

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finnews_sentiment.models.evaluate import evaluate_baselines, evaluate_predictions


def test_known_metrics_match_independent_hand_calculation() -> None:
    truth = [0, 0, 1, 1, 2, 2, 2]
    predictions = [0, 1, 1, 1, 0, 2, 2]

    metrics = evaluate_predictions(truth, predictions)

    assert metrics.n_samples == 7
    assert metrics.accuracy == pytest.approx(5 / 7)
    assert metrics.macro_f1 == pytest.approx((0.5 + 0.8 + 0.8) / 3)
    assert [(item.class_id, item.class_name, item.support) for item in metrics.per_class] == [
        (0, "negative", 2), (1, "neutral", 2), (2, "positive", 3),
    ]
    assert [item.precision for item in metrics.per_class] == pytest.approx([0.5, 2 / 3, 1.0])
    assert [item.recall for item in metrics.per_class] == pytest.approx([0.5, 1.0, 2 / 3])
    assert [item.f1 for item in metrics.per_class] == pytest.approx([0.5, 0.8, 0.8])
    assert sum(item.support for item in metrics.per_class) == metrics.n_samples


def test_missing_predicted_classes_remain_with_zero_metrics() -> None:
    metrics = evaluate_predictions([0, 1, 2], [1, 1, 1])

    assert metrics.n_samples == 3
    assert metrics.accuracy == pytest.approx(1 / 3)
    assert metrics.macro_f1 == pytest.approx(1 / 6)
    assert [item.f1 for item in metrics.per_class] == pytest.approx([0, 0.5, 0])
    assert [item.support for item in metrics.per_class] == [1, 1, 1]
    assert sum(item.support for item in metrics.per_class) == 3


def test_baseline_comparison_uses_same_rows_and_fixed_metric_columns() -> None:
    result = evaluate_baselines(
        np.asarray([0, 1, 2]), np.asarray([0, 1, 1]), np.asarray([1, 1, 1]),
    )

    assert result.baseline.n_samples == result.dummy.n_samples == 3
    assert result.comparison["model"].tolist() == ["baseline", "dummy"]
    assert result.comparison["n_samples"].tolist() == [3, 3]
    assert "macro_f1" in result.comparison.columns
    assert "negative_support" in result.comparison.columns
    assert "neutral_recall" in result.comparison.columns
    assert "positive_f1" in result.comparison.columns


@pytest.mark.parametrize("values", [[], np.asarray([], dtype=np.int64), [[0, 1]], [0.0, 1.0], [False, True]])
def test_invalid_prediction_vectors_fail(values: object) -> None:
    with pytest.raises(ValueError):
        evaluate_predictions(cast(Sequence[int], values), [0, 1])


@pytest.mark.parametrize("bad_id", [-1, 3, "1"])
def test_unknown_class_ids_fail(bad_id: object) -> None:
    with pytest.raises(ValueError, match="unknown class ID|integer class IDs"):
        evaluate_predictions([0, 1], cast(Sequence[int], [0, bad_id]))


def test_different_lengths_fail_for_individual_and_comparison_metrics() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        evaluate_predictions([0, 1], [0])
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_baselines([0, 1], [0, 1], [1])
