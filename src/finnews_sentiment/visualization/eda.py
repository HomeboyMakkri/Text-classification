"""Read-only EDA for raw financial sentiment data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from finnews_sentiment.data.load_data import CLASS_ORDER, validate_news


def quality_summary(df: pd.DataFrame) -> dict[str, object]:
    """Count raw rows; duplicates are counted beyond their first occurrence."""
    validate_news(df)
    text = df["text"].astype("string")
    valid_text = text.notna() & text.str.strip().ne("").fillna(False)
    labeled = df.loc[valid_text & df["sentiment"].notna()]
    label_counts = labeled.groupby("text")["sentiment"].nunique()
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {name: str(dtype) for name, dtype in df.dtypes.items()},
        "missing": {name: int(count) for name, count in df.isna().sum().items()},
        "blank_texts": int((text.notna() & text.str.strip().eq("").fillna(False)).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_texts": int(text.loc[valid_text].duplicated().sum()),
        "conflicting_texts": int((label_counts > 1).sum()),
    }


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed class order; fractions use all rows, including missing labels."""
    validate_news(df)
    counts = df["sentiment"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    return pd.DataFrame({"count": counts, "fraction": counts / len(df)}).rename_axis("sentiment")


def text_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Characters and whitespace-separated words, not model tokenizer tokens."""
    validate_news(df)
    text = df["text"].astype("string")
    return pd.DataFrame({
        "char_count": text.str.len().astype("Int64"),
        "word_count": text.map(lambda value: len(value.split()) if isinstance(value, str) else pd.NA).astype("Int64"),
    }, index=df.index)


def class_examples(df: pd.DataFrame, n: int = 3) -> dict[str, list[str]]:
    """Take deterministic first nonblank examples; absent classes return []."""
    validate_news(df)
    if n < 1:
        raise ValueError("n must be positive")
    text = df["text"].astype("string")
    valid = text.notna() & text.str.strip().ne("").fillna(False)
    return {
        sentiment: df.loc[valid & df["sentiment"].eq(sentiment), "text"].head(n).tolist()
        for sentiment in CLASS_ORDER
    }


def plot_class_distribution(df: pd.DataFrame, path: Path | None = None) -> Figure:
    counts = class_distribution(df)["count"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.to_numpy())
    ax.set(title="Financial sentiment classes (raw data)", ylabel="Rows", xlabel="Sentiment")
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_text_length_distribution(
    df: pd.DataFrame, path: Path | None = None, bins: int = 50
) -> Figure:
    lengths = text_lengths(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column, label in zip(axes, lengths.columns, ("Characters", "Whitespace-separated words")):
        ax.hist(lengths[column].dropna().to_numpy(dtype=float), bins=bins)
        ax.set(xlabel=label, ylabel="Rows")
    fig.suptitle("Text length distributions (raw data)")
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig
