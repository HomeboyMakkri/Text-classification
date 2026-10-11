"""Minimal text cleanup; table/feature helpers below remain curriculum templates."""

from html import unescape
from html.entities import html5
import re

import pandas as pd


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


def clean_news_data(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Применяет preprocess_text ко всему DataFrame.

    Создаёт новую колонку `text_clean`.
    """
    df = df.copy()
    df["text_clean"] = df[text_col].astype(str).map(preprocess_text)
    return df


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
