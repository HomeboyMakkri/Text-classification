"""Pytest checks for fixed class IDs and duplicate/conflict boundaries."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finnews_sentiment.features.labels_duplicates import (
    DuplicateLabelConflictError, add_label_ids, analyze_duplicates, prepare_labeled_news,
)
from finnews_sentiment.features.preprocess import preprocess_news


@pytest.mark.parametrize("labels,expected", [
    (["positive", "negative", "neutral"], [2, 0, 1]),
    (["positive", "positive"], [2, 2]),
    (["neutral", "negative"], [1, 0]),
])
def test_mapping_independent_of_order_and_missing_classes(labels: list[str], expected: list[int]) -> None:
    raw = pd.DataFrame({"sentiment": labels}, index=[7] * len(labels))
    snapshot = raw.copy(deep=True)
    result = add_label_ids(raw)
    assert result["label_id"].tolist() == expected
    assert str(result["label_id"].dtype) == "int64"
    assert result["sentiment"].tolist() == labels
    pd.testing.assert_frame_equal(raw, snapshot)


@pytest.mark.parametrize("label", [None, pd.NA, "bullish", "", 2])
def test_invalid_labels_have_no_fallback(label: object) -> None:
    with pytest.raises(ValueError, match="Invalid sentiment"):
        add_label_ids(pd.DataFrame({"sentiment": [label]}))


def test_raw_and_new_clean_groups_preserve_rows_and_order() -> None:
    raw = pd.DataFrame({
        "text": ["B", "A", "B", "a", "A", "unique"],
        "sentiment": ["neutral"] * 6,
    })
    processed, _ = preprocess_news(raw)
    snapshot = processed.copy(deep=True)
    labeled, report = prepare_labeled_news(processed)
    assert len(labeled) == 6
    assert labeled["source_row_id"].tolist() == list(range(6))
    assert [group.source_row_ids for group in report.raw_groups] == [(0, 2), (1, 4)]
    assert [group.source_row_ids for group in report.clean_groups] == [(0, 2), (1, 3, 4)]
    assert [group.new_after_cleaning for group in report.clean_groups] == [False, True]
    assert report.raw_duplicate_rows == 2
    assert report.clean_duplicate_rows == 3
    assert not report.has_conflicts
    assert analyze_duplicates(processed) == report
    pd.testing.assert_frame_equal(processed, snapshot)


@pytest.mark.parametrize("texts,raw_conflict", [
    (["Profit", "Profit"], True),
    (["<b>Profit</b>", "profit"], False),
])
def test_conflicts_are_diagnostic_and_block_final_preparation(texts: list[str], raw_conflict: bool) -> None:
    processed, _ = preprocess_news(pd.DataFrame({
        "text": texts, "sentiment": ["negative", "positive"],
    }))
    snapshot = processed.copy(deep=True)
    report = analyze_duplicates(processed)
    assert report.has_conflicts
    assert any(group.conflicting for group in report.raw_groups) == raw_conflict
    assert report.clean_groups[0].sentiments == ("negative", "positive")
    with pytest.raises(DuplicateLabelConflictError) as caught:
        prepare_labeled_news(processed)
    assert caught.value.report == report
    pd.testing.assert_frame_equal(processed, snapshot)


def test_source_ids_with_gaps_are_retained() -> None:
    processed, _ = preprocess_news(pd.DataFrame({
        "text": ["Same", None, "SAME"], "sentiment": ["neutral"] * 3,
    }))
    labeled, report = prepare_labeled_news(processed)
    assert labeled["source_row_id"].tolist() == [0, 2]
    assert report.clean_groups[0].source_row_ids == (0, 2)
    assert list(labeled.columns) == ["source_row_id", "text", "sentiment", "text_clean", "label_id"]


@pytest.mark.parametrize("column,values", [
    ("source_row_id", [0, 0]),
    ("source_row_id", [-1, 1]),
    ("source_row_id", [0.5, 1.5]),
    ("source_row_id", [False, True]),
    ("text_clean", ["", "news"]),
    ("text_clean", [None, "news"]),
    ("text", [42, "news"]),
])
def test_malformed_processed_input(column: str, values: list[object]) -> None:
    processed, _ = preprocess_news(pd.DataFrame({
        "text": ["A", "B"], "sentiment": ["neutral", "positive"],
    }))
    processed[column] = pd.Series(values)
    with pytest.raises(ValueError):
        analyze_duplicates(processed)


def test_duplicate_analysis_requires_processed_schema() -> None:
    with pytest.raises(ValueError, match="Unique columns"):
        analyze_duplicates(pd.DataFrame({"text": ["news"], "sentiment": ["neutral"]}))
