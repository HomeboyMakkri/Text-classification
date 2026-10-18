"""Asymmetric confusion fixtures expose transposition and denominator errors."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.models.confusion import (
    build_validation_confusion, plot_validation_confusion, save_validation_confusion,
)
from finnews_sentiment.data.load_data import CLASS_ORDER


def _table(counts: list[list[int]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"true_label_id": i, "pred_label_id": j}
        for i, row in enumerate(counts) for j, n in enumerate(row) for _ in range(n)
    ])


def test_asymmetric_matrix_and_all_denominators() -> None:
    expected = np.array([[4, 2, 1], [3, 5, 0], [0, 4, 6]])
    table = _table(expected.tolist())
    before = table.copy(deep=True)
    result = build_validation_confusion(table)
    np.testing.assert_array_equal(result.counts.to_numpy(), expected)
    assert result.counts.index.tolist() == list(CLASS_ORDER)
    assert result.counts.columns.tolist() == list(CLASS_ORDER)
    assert result.counts.index.name == "true_label"
    assert result.counts.columns.name == "pred_label"
    assert all(pd.api.types.is_integer_dtype(dtype) for dtype in result.counts.dtypes)
    assert expected.sum() == result.metrics.n_samples == len(table)
    assert np.trace(expected) == int(table["true_label_id"].eq(table["pred_label_id"]).sum())
    assert result.pairs["count"].sum() == expected.sum() - np.trace(expected)
    assert len(result.pairs) == 6
    np.testing.assert_array_equal(result.counts.sum(axis=1), result.per_class["support"])
    np.testing.assert_array_equal(result.counts.sum(axis=0), result.per_class["predicted_count"])
    np.testing.assert_allclose(result.normalized.to_numpy(), expected / expected.sum(axis=1)[:, None])
    np.testing.assert_allclose(np.diag(result.normalized), result.per_class["recall"])
    np.testing.assert_allclose(result.per_class["precision"], np.diag(expected) / expected.sum(axis=0))
    pair = result.pairs.iloc[0]
    assert (pair["true_label"], pair["pred_label"], pair["count"], pair["true_support"]) == ("negative", "neutral", 2, 7)
    assert pair["fraction_of_true"] == pytest.approx(2 / 7)
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("counts", [
    [[0, 0, 0], [2, 3, 0], [0, 4, 0]],  # absent true negative, absent predicted positive
    [[2, 0, 0], [0, 3, 0], [0, 0, 4]],  # no errors
    [[0, 2, 0], [0, 0, 0], [0, 3, 0]],  # only one predicted class
])
def test_absent_classes_and_no_errors(counts: list[list[int]]) -> None:
    result = build_validation_confusion(_table(counts))
    assert result.counts.shape == result.normalized.shape == (3, 3)
    assert len(result.pairs) == 6
    assert np.isfinite(result.normalized.to_numpy()).all()
    for i, row in enumerate(counts):
        assert result.normalized.iloc[i].sum() == pytest.approx(1 if sum(row) else 0)
        if sum(row) == 0:
            assert result.normalized.iloc[i].eq(0).all()
            assert result.per_class.iloc[i]["recall"] == 0
    assert result.pairs["count"].sum() == sum(sum(row) - row[i] for i, row in enumerate(counts))


def test_headless_exports_and_absent_annotation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    result = build_validation_confusion(_table([[0, 0, 0], [1, 2, 0], [0, 1, 1]]))
    figure = plot_validation_confusion(result, normalized=True)
    axis = figure.axes[0]
    assert axis.get_xlabel() == "Predicted label"
    assert axis.get_ylabel() == "True label"
    assert "support=0" in axis.get_yticklabels()[0].get_text()
    assert "absent" in axis.get_yticklabels()[0].get_text()
    assert "true support" in axis.get_title()
    assert axis.texts[3].get_text() == "33.3%"
    paths = save_validation_confusion(result, tmp_path / "reports")
    assert len(paths) == 5
    for name in ("counts", "normalized"):
        loaded = pd.read_csv(paths[f"{name}_csv"], index_col="true_label")
        loaded.columns.name = "pred_label"
        pd.testing.assert_frame_equal(loaded, getattr(result, name))
        assert paths[f"{name}_png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    pd.testing.assert_frame_equal(pd.read_csv(paths["pairs_csv"]), result.pairs)


@pytest.mark.parametrize("table", [
    pd.DataFrame({"true_label_id": [], "pred_label_id": []}),
    pd.DataFrame({"true_label_id": [0]}),
    pd.DataFrame({"true_label_id": [0], "pred_label_id": [3]}),
    pd.DataFrame({"true_label_id": [0], "pred_label_id": [None]}),
    pd.DataFrame({"true_label_id": [0], "pred_label_id": [1.5]}),
])
def test_invalid_labels_use_shared_metric_contract(table: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        build_validation_confusion(table)
