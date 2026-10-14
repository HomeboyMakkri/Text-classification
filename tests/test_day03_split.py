"""Leakage, row preservation, reproducibility and stratification boundaries."""

from itertools import combinations
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.data.split_data import SPLIT_ORDER, split_processed_news
from finnews_sentiment.features.labels_duplicates import DuplicateLabelConflictError


@pytest.fixture
def news() -> pd.DataFrame:
    rows = [
        (i * 3, f"News {i}", CLASS_ORDER[i % 3], f"news {i}", i % 3)
        for i in range(45)
    ]
    rows.extend([
        (135, "News 0", "negative", "news 0", 0),
        (138, "NEWS 0", "negative", "news 0", 0),
        (141, "News 1", "neutral", "news 1", 1),
    ])
    return pd.DataFrame(rows, columns=["source_row_id", "text", "sentiment", "text_clean", "label_id"], index=[7] * len(rows))


def test_coverage_order_groups_balance_and_immutability(news: pd.DataFrame) -> None:
    before = news.copy(deep=True)
    result = split_processed_news(news)
    pd.testing.assert_frame_equal(news, before)
    assert list(result.manifest.columns) == ["source_row_id", "group_id", "split"]
    assert result.manifest["source_row_id"].tolist() == news["source_row_id"].tolist()
    assert result.manifest["source_row_id"].is_unique
    assert result.manifest["group_id"].nunique() == 45
    assert result.summary.rows == 48 and result.summary.groups == 45
    assert result.summary.algorithm == "group-stratified-v1"
    assert result.summary.random_state == 42
    assert result.summary.disjoint == {"source_row_id": True, "text": True, "text_clean": True}
    for name in SPLIT_ORDER:
        frame, stats = result.splits[name], result.summary.splits[name]
        ids = result.manifest.loc[result.manifest["split"] == name, "source_row_id"].tolist()
        pd.testing.assert_frame_equal(frame, news.loc[news["source_row_id"].isin(ids)])
        assert frame["source_row_id"].is_monotonic_increasing
        assert set(frame["sentiment"]) == set(CLASS_ORDER)
        assert sum(stats.row_class_counts.values()) == stats.rows == len(frame)
        assert sum(stats.group_class_counts.values()) == stats.groups == frame["text_clean"].nunique()
        assert stats.row_share == len(frame) / 48
        assert stats.group_share == stats.groups / 45
        assert sum(stats.row_class_shares.values()) == pytest.approx(1)
        assert sum(stats.group_class_shares.values()) == pytest.approx(1)
        for label in CLASS_ORDER:
            assert stats.row_class_counts[label] == int(frame["sentiment"].eq(label).sum())
            assert stats.group_class_counts[label] == frame.loc[frame['sentiment'] == label, 'text_clean'].nunique()
            assert stats.row_class_shares[label] == stats.row_class_counts[label] / stats.rows
            assert stats.group_class_shares[label] == stats.group_class_counts[label] / stats.groups
    for left, right in combinations(SPLIT_ORDER, 2):
        for column in ("source_row_id", "text", "text_clean"):
            assert set(result.splits[left][column]).isdisjoint(result.splits[right][column])
    assert sum(stats.row_share for stats in result.summary.splits.values()) == pytest.approx(1)
    assert sum(stats.group_share for stats in result.summary.splits.values()) == pytest.approx(1)
    assert [result.summary.splits[name].groups for name in SPLIT_ORDER] == [28, 8, 9]
    duplicates = result.manifest.loc[result.manifest['source_row_id'].isin([0, 135, 138])]
    assert duplicates['group_id'].tolist() == [0, 0, 0]
    assert duplicates['split'].nunique() == 1


def test_repeat_and_pandas_index_independence(news: pd.DataFrame) -> None:
    first = split_processed_news(news)
    repeated = split_processed_news(news)
    changed_index = news.copy(deep=True)
    changed_index.index = pd.Index(range(900, 948))
    reindexed = split_processed_news(changed_index)
    pd.testing.assert_frame_equal(first.manifest, repeated.manifest)
    pd.testing.assert_frame_equal(first.manifest, reindexed.manifest)
    assert first.summary == repeated.summary == reindexed.summary


def test_repeated_rows_do_not_change_group_assignment(news: pd.DataFrame) -> None:
    full = split_processed_news(news)
    unique = split_processed_news(news.iloc[:45])
    pd.testing.assert_frame_equal(full.manifest.iloc[:45], unique.manifest)
    assert any(full.summary.splits[name].row_share != full.summary.splits[name].group_share for name in SPLIT_ORDER)


@pytest.mark.parametrize('clean_only', [False, True])
def test_conflicting_groups_are_not_relabelled(news: pd.DataFrame, clean_only: bool) -> None:
    frame = news.copy(deep=True)
    frame.loc[frame['source_row_id'] == 135, 'sentiment'] = 'positive'
    frame.loc[frame['source_row_id'] == 135, 'label_id'] = 2
    if clean_only:
        frame.loc[frame['source_row_id'] == 135, 'text'] = 'Another raw spelling'
    with pytest.raises(DuplicateLabelConflictError):
        split_processed_news(frame)


def test_inconsistent_raw_to_clean_fails(news: pd.DataFrame) -> None:
    frame = news.copy(deep=True)
    frame.loc[frame['source_row_id'] == 135, 'text_clean'] = 'inconsistent'
    with pytest.raises(ValueError, match='inconsistent text_clean'):
        split_processed_news(frame)


@pytest.mark.parametrize('counts,stage', [([1, 15, 15], 'test'), ([2, 2, 2], 'test'), ([4, 4, 4], 'validation')])
def test_small_group_classes_fail_with_diagnostics(counts: list[int], stage: str) -> None:
    rows = [(f'item {label} {i}', CLASS_ORDER[label], label) for label, n in enumerate(counts) for i in range(n)]
    frame = pd.DataFrame({
        'source_row_id': range(len(rows)), 'text': [r[0] for r in rows],
        'sentiment': [r[1] for r in rows], 'text_clean': [r[0] for r in rows], 'label_id': [r[2] for r in rows],
    })
    with pytest.raises(ValueError, match=f'Cannot stratify {stage}.*row class counts=.*group class counts='):
        split_processed_news(frame)


@pytest.mark.parametrize('key,value', [
    ('test_size', 0), ('test_size', 1), ('test_size', True), ('test_size', float('nan')),
    ('validation_size', -0.2), ('validation_size', float('inf')),
    ('random_state', -1), ('random_state', True), ('random_state', 2**32),
])
def test_invalid_settings(news: pd.DataFrame, key: str, value: float | int) -> None:
    if key == 'test_size':
        with pytest.raises(ValueError, match=key):
            split_processed_news(news, test_size=value)
    elif key == 'validation_size':
        with pytest.raises(ValueError, match=key):
            split_processed_news(news, validation_size=value)
    else:
        with pytest.raises(ValueError, match=key):
            split_processed_news(news, random_state=int(value) if not isinstance(value, bool) else value)


def test_seed42_two_stage_reference(news: pd.DataFrame) -> None:
    # Reference membership for 45 interleaved groups in pinned sklearn runtime.
    # This also checks sorting outer-train before validation stratification.
    manifest = split_processed_news(news.iloc[:45]).manifest
    assert manifest.loc[manifest['split'] == 'test', 'group_id'].tolist() == [21, 27, 51, 54, 66, 87, 105, 108, 129]
    assert manifest.loc[manifest['split'] == 'validation', 'group_id'].tolist() == [30, 60, 81, 84, 114, 117, 120, 132]


def test_missing_class_fails_before_split(news: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match='All three classes'):
        split_processed_news(news.loc[news['sentiment'] != 'negative'])


def test_rounding_that_drops_a_class_has_no_fallback() -> None:
    labels = [0] * 3 + [1] * 15 + [2] * 15
    text = [f'row {i}' for i in range(len(labels))]
    frame = pd.DataFrame({
        'source_row_id': range(len(labels)), 'text': text,
        'sentiment': [CLASS_ORDER[label] for label in labels],
        'text_clean': text, 'label_id': labels,
    })
    with pytest.raises(ValueError, match='All three classes required in validation.*split group counts='):
        split_processed_news(frame)
