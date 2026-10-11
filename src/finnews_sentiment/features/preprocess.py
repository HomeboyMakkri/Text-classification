"""Minimal text/table cleanup; add_basic_features remains a curriculum template."""

from dataclasses import dataclass
from html import unescape
from html.entities import html5
import re
import warnings

import pandas as pd

from finnews_sentiment.data.load_data import CLASS_ORDER, validate_news

PREPROCESSING_POLICY = "minimal-v1"
_ENTITY_PATTERN = re.compile(r"&(?:\#[0-9]+;?|\#[xX][0-9a-fA-F]+;?|[A-Za-z][A-Za-z0-9]*;?)")
# Match complete tags, including quoted attributes containing '>'. Comparisons
# and incomplete markup do not match. This is text cleanup, not a web scraper.
_MARKUP_PATTERN = re.compile(
    r'''<!--.*?-->
    | <(?:
        [A-Za-z][A-Za-z0-9:-]*
        (?:\s+[^\s/<>="'`]+
            (?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s<>="'`]+))?
        )*\s*/?
        | /[A-Za-z][A-Za-z0-9:-]*\s*
    )>''',
    re.VERBOSE | re.DOTALL,
)

TICKER_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")  # грубое приближение тикеров


def _decode_entity(match: re.Match[str]) -> str:
    entity = match.group(0)
    # html.unescape alone can decode a known prefix of an unknown entity
    # (e.g. &notareal;). Only decode complete known names or numeric references.
    if entity.startswith("&#") or entity[1:] in html5:
        return unescape(entity)
    return entity


def preprocess_text(text: str) -> str:
    """Apply minimal-v1: entities, markup, lowercase and whitespace to stability.

    Preserve negation, numbers, currencies and punctuation. Empty strings stay
    empty; non-strings raise TypeError. No fitted state or NLTK resources needed.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    while True:
        decoded = _ENTITY_PATTERN.sub(_decode_entity, text)
        without_markup = _MARKUP_PATTERN.sub(" ", decoded)
        normalized = " ".join(without_markup.lower().split())
        if normalized == text:
            return normalized
        text = normalized


@dataclass(frozen=True)
class ExcludedRow:
    source_row_id: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessingReport:
    policy: str
    n_input: int
    n_retained: int
    n_excluded: int
    excluded_rows: tuple[ExcludedRow, ...]
    reason_counts: dict[str, int]
    class_counts_before: dict[str, int]
    class_counts_after: dict[str, int]
    missing_sentiment_before: int
    changed_source_row_ids: tuple[int, ...]


class EmptyProcessedDatasetError(ValueError):
    """No usable rows; retain the report for inspection by the caller."""

    def __init__(self, report: PreprocessingReport) -> None:
        self.report = report
        super().__init__(
            f"All {report.n_input} rows excluded; reasons={report.reason_counts}"
        )


def preprocess_news(df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Validate canonical rows, clean text and report every excluded source ID.

    IDs refer to input positions, not index labels. The caller must supply the
    canonical CSV in its original order. Keep duplicates and original values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not df.columns.is_unique:
        raise ValueError("DataFrame column names must be unique")
    validate_news(df)
    retained: list[dict[str, str | int]] = []
    excluded: list[ExcludedRow] = []
    changed_ids: list[int] = []
    reason_counts: dict[str, int] = {
        reason: 0 for reason in
        ("missing_text", "missing_sentiment", "blank_text", "empty_after_cleaning")
    }
    before: dict[str, int] = {label: 0 for label in CLASS_ORDER}
    after: dict[str, int] = {label: 0 for label in CLASS_ORDER}
    missing_sentiment = 0
    for position, (text, sentiment) in enumerate(
        zip(df["text"], df["sentiment"], strict=True)
    ):
        reasons: list[str] = []
        cleaned = ""
        if isinstance(sentiment, str):
            before[sentiment] += 1
        else:
            missing_sentiment += 1
            reasons.append("missing_sentiment")
        if not isinstance(text, str):
            reasons.insert(0, "missing_text")
        elif not text.strip():
            reasons.append("blank_text")
        else:
            cleaned = preprocess_text(text)
            if not cleaned:
                reasons.append("empty_after_cleaning")
        if reasons:
            excluded.append(ExcludedRow(position, tuple(reasons)))
            for reason in reasons:
                reason_counts[reason] += 1
            continue
        # validate_news established that all non-missing values are strings.
        if not isinstance(text, str) or not isinstance(sentiment, str):
            raise TypeError("Retained text and sentiment must be strings")
        retained.append({
            "source_row_id": position, "text": text,
            "sentiment": sentiment, "text_clean": cleaned,
        })
        after[sentiment] += 1
        if cleaned != text:
            changed_ids.append(position)
    report = PreprocessingReport(
        policy=PREPROCESSING_POLICY, n_input=len(df), n_retained=len(retained),
        n_excluded=len(excluded), excluded_rows=tuple(excluded),
        reason_counts=reason_counts, class_counts_before=before,
        class_counts_after=after, missing_sentiment_before=missing_sentiment,
        changed_source_row_ids=tuple(changed_ids),
    )
    if not retained:
        raise EmptyProcessedDatasetError(report)
    processed = pd.DataFrame(
        retained, columns=["source_row_id", "text", "sentiment", "text_clean"]
    )
    return processed, report


def clean_news_data(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Compatibility caller; use preprocess_news to also inspect exclusions.

    Canonical text column is 'text'; a custom input column is renamed explicitly.
    """
    if text_col != "text":
        if text_col not in df.columns:
            raise ValueError(f"Missing text column: {text_col!r}")
        if "text" in df.columns:
            raise ValueError("Custom text column conflicts with canonical 'text'")
        df = df.rename(columns={text_col: "text"})
    processed, report = preprocess_news(df)
    if report.n_excluded:
        warnings.warn(
            f"Excluded {report.n_excluded} rows: {report.reason_counts}; "
            "use preprocess_news for source IDs and full report",
            UserWarning, stacklevel=2,
        )
    return processed


def add_basic_features(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    """Добавляет простые числовые признаки на основе текста.

    Примеры:
    - длина текста в словах
    - количество тикеров
    - количество чисел в исходном тексте
    """
    df = df.copy()
    df["text_len_words"] = df[text_col].astype(str).str.split().apply(len)
    df["ticker_count"] = df["text"].astype(str).apply(lambda s: len(TICKER_PATTERN.findall(s)))

    # количество чисел в исходном тексте
    df["number_count"] = df["text"].astype(str).str.findall(r"\d+(?:\.\d+)?").apply(len)

    return df
