"""Fixed category IDs and diagnostic duplicate groups; no deletion or splitting."""

from dataclasses import dataclass
from numbers import Integral

import pandas as pd

from finnews_sentiment.data.load_data import CLASS_ORDER

LABEL_MAPPING: dict[str, int] = {label: index for index, label in enumerate(CLASS_ORDER)}


@dataclass(frozen=True)
class DuplicateGroup:
    text: str
    source_row_ids: tuple[int, ...]
    sentiments: tuple[str, ...]
    new_after_cleaning: bool

    @property
    def conflicting(self) -> bool:
        return len(set(self.sentiments)) > 1


@dataclass(frozen=True)
class DuplicateReport:
    raw_groups: tuple[DuplicateGroup, ...]
    clean_groups: tuple[DuplicateGroup, ...]

    @property
    def raw_duplicate_rows(self) -> int:
        """Repeated rows beyond the first per raw text, irrespective of label."""
        return sum(len(group.source_row_ids) - 1 for group in self.raw_groups)

    @property
    def clean_duplicate_rows(self) -> int:
        return sum(len(group.source_row_ids) - 1 for group in self.clean_groups)

    @property
    def has_conflicts(self) -> bool:
        return any(group.conflicting for group in (*self.raw_groups, *self.clean_groups))


class DuplicateLabelConflictError(ValueError):
    """Final preparation cannot proceed; the full diagnostic report is retained."""

    def __init__(self, report: DuplicateReport) -> None:
        self.report = report
        super().__init__(
            "Conflicting labels in raw/clean text groups; inspect .report "
            "and resolve explicitly before final preparation"
        )


def add_label_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Copy rows and add integer category IDs, independent of observed classes."""
    if not df.columns.is_unique or "sentiment" not in df.columns:
        raise ValueError("Unique columns including sentiment are required")
    if df.empty:
        raise ValueError("Dataset contains no rows")
    for sentiment in df["sentiment"]:
        if not isinstance(sentiment, str) or sentiment not in LABEL_MAPPING:
            raise ValueError(f"Invalid sentiment {sentiment!r}; expected {CLASS_ORDER}")
    result = df.copy(deep=True)
    result["label_id"] = pd.Series(
        [LABEL_MAPPING[sentiment] for sentiment in df["sentiment"]],
        index=df.index, dtype="int64",
    )
    return result


def analyze_duplicates(df: pd.DataFrame) -> DuplicateReport:
    """Inspect retained D2-03 rows. Group/member order follows input row order."""
    required = {"source_row_id", "text", "sentiment", "text_clean"}
    if not df.columns.is_unique or not required.issubset(df.columns):
        raise ValueError(f"Unique columns including {sorted(required)} are required")
    add_label_ids(df)  # Validate the fixed class contract without modifying input.
    if not df["source_row_id"].is_unique:
        raise ValueError("source_row_id values must be unique")
    members: list[tuple[int, str, str, str]] = []
    for source_id, raw, sentiment, clean in zip(
        df["source_row_id"], df["text"], df["sentiment"], df["text_clean"], strict=True
    ):
        if not isinstance(source_id, Integral) or isinstance(source_id, bool) or source_id < 0:
            raise ValueError("source_row_id must contain nonnegative integers")
        if not isinstance(raw, str) or not raw.strip() or not isinstance(clean, str) or not clean.strip():
            raise ValueError("text and text_clean must contain nonblank strings")
        if not isinstance(sentiment, str):
            raise ValueError("sentiment must contain strings")
        members.append((int(source_id), raw, sentiment, clean))
    groups: list[tuple[DuplicateGroup, ...]] = []
    for use_clean in (False, True):
        buckets: dict[str, list[tuple[int, str, str, str]]] = {}
        for member in members:
            key = member[3] if use_clean else member[1]
            buckets.setdefault(key, []).append(member)
        groups.append(tuple(
            DuplicateGroup(
                text=text, source_row_ids=tuple(row[0] for row in rows),
                sentiments=tuple(row[2] for row in rows),
                new_after_cleaning=use_clean and len({row[1] for row in rows}) > 1,
            )
            for text, rows in buckets.items() if len(rows) > 1
        ))
    return DuplicateReport(raw_groups=groups[0], clean_groups=groups[1])


def prepare_labeled_news(df: pd.DataFrame) -> tuple[pd.DataFrame, DuplicateReport]:
    """Finalize D2-03 rows with IDs; block unresolved label conflicts."""
    report = analyze_duplicates(df)
    if report.has_conflicts:
        raise DuplicateLabelConflictError(report)
    return add_label_ids(df), report
