"""Deterministic stratification of exact clean-text groups; no fitting or I/O."""

from dataclasses import dataclass
from itertools import combinations
import math

import pandas as pd
from sklearn.model_selection import train_test_split

from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.data.load_processed import validate_processed_news

SPLIT_ORDER = ("train", "validation", "test")
SPLIT_ALGORITHM = "group-stratified-v1"


@dataclass(frozen=True)
class SplitStats:
    rows: int
    groups: int
    row_share: float
    group_share: float
    row_class_counts: dict[str, int]
    group_class_counts: dict[str, int]
    row_class_shares: dict[str, float]
    group_class_shares: dict[str, float]


@dataclass(frozen=True)
class SplitSummary:
    algorithm: str
    random_state: int
    test_size: float
    validation_size: float
    rows: int
    groups: int
    splits: dict[str, SplitStats]
    disjoint: dict[str, bool]


@dataclass(frozen=True)
class NewsSplits:
    manifest: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    summary: SplitSummary


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["sentiment"].value_counts()
    return {label: int(counts.get(label, 0)) for label in CLASS_ORDER}


def _fraction(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be a finite fraction strictly between 0 and 1")


def split_processed_news(
    df: pd.DataFrame, *, test_size: float = 0.2,
    validation_size: float = 0.2, random_state: int = 42,
) -> NewsSplits:
    """Split unique groups twice, then preserve every row in canonical order.

    validation_size is a fraction of outer-train groups. group_id is the
    minimum source_row_id, independent of the pandas index. Fractions concern
    groups; repeated rows can change row proportions. Invalid/small inputs
    raise without falling back to a row split, another seed or deduplication.
    """
    validate_processed_news(df)
    _fraction(test_size, "test_size")
    _fraction(validation_size, "validation_size")
    if isinstance(random_state, bool) or not isinstance(random_state, int) or not 0 <= random_state < 2**32:
        raise ValueError("random_state must be an integer between 0 and 2**32 - 1")

    # The validator requires increasing source IDs, so first occurrence is min.
    groups = df.loc[~df["text_clean"].duplicated()].copy()
    group_ids = [int(value) for value in groups["source_row_id"]]
    labels = {group_id: int(label) for group_id, label in zip(group_ids, groups["label_id"], strict=True)}
    source_by_clean = dict(zip(groups["text_clean"], group_ids, strict=True))
    diagnostic = f"row class counts={_class_counts(df)}, group class counts={_class_counts(groups)}"

    def stratify(ids: list[int], fraction: float, stage: str) -> tuple[list[int], list[int]]:
        stage_counts = {name: sum(labels[group_id] == label for group_id in ids) for label, name in enumerate(CLASS_ORDER)}
        try:
            left, right = train_test_split(
                ids, test_size=fraction, random_state=random_state,
                stratify=[labels[group_id] for group_id in ids],
            )
        except ValueError as error:
            raise ValueError(
                f"Cannot stratify {stage}: {diagnostic}; stage group counts={stage_counts}. {error}"
            ) from error
        # Sorting outer-train before the second split is part of the algorithm.
        return sorted(left), sorted(right)

    outer_train, test = stratify(group_ids, test_size, "test")
    train, validation = stratify(outer_train, validation_size, "validation")
    assignment = {group_id: split for split, ids in zip(SPLIT_ORDER, (train, validation, test), strict=True) for group_id in ids}
    row_groups = [source_by_clean[text] for text in df["text_clean"]]
    row_splits = [assignment[group_id] for group_id in row_groups]
    manifest = pd.DataFrame({
        "source_row_id": df["source_row_id"].tolist(),
        "group_id": row_groups, "split": row_splits,
    })
    splits = {split: df.loc[[value == split for value in row_splits]].copy(deep=True) for split in SPLIT_ORDER}
    stats: dict[str, SplitStats] = {}
    for split, frame in splits.items():
        unique = frame.loc[~frame["text_clean"].duplicated()]
        row_counts, group_counts = _class_counts(frame), _class_counts(unique)
        if any(count == 0 for count in group_counts.values()):
            raise ValueError(f"All three classes required in {split}: {diagnostic}; split group counts={group_counts}")
        stats[split] = SplitStats(
            rows=len(frame), groups=len(unique), row_share=len(frame) / len(df),
            group_share=len(unique) / len(groups), row_class_counts=row_counts,
            group_class_counts=group_counts,
            row_class_shares={label: count / len(frame) for label, count in row_counts.items()},
            group_class_shares={label: count / len(unique) for label, count in group_counts.items()},
        )
    disjoint = {
        column: all(
            set(splits[left][column]).isdisjoint(splits[right][column])
            for left, right in combinations(SPLIT_ORDER, 2)
        ) for column in ("source_row_id", "text", "text_clean")
    }
    if not all(disjoint.values()) or sum(len(frame) for frame in splits.values()) != len(df):
        raise ValueError("Split invariant violated: incomplete coverage or overlapping source IDs/raw/clean texts")
    return NewsSplits(manifest, splits, SplitSummary(
        algorithm=SPLIT_ALGORITHM, random_state=random_state, test_size=test_size,
        validation_size=validation_size, rows=len(df), groups=len(groups),
        splits=stats, disjoint=disjoint,
    ))
