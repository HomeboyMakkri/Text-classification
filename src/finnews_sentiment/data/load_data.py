"""Canonical CSV loading: preserve raw rows and missing values for inspection."""

import argparse
from pathlib import Path

import pandas as pd

CLASS_ORDER = ("negative", "neutral", "positive")


def validate_news(df: pd.DataFrame) -> None:
    """Check schema without cleaning away evidence needed for EDA."""
    missing_columns = {"text", "sentiment"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError("Dataset contains no rows")
    for column in ("text", "sentiment"):
        if not df[column].dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Column {column!r} must contain strings or missing values")
    unknown = set(df["sentiment"].dropna()) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(f"Unknown sentiment labels: {sorted(unknown)}; expected {CLASS_ORDER}")


def load_raw_news(path: Path) -> pd.DataFrame:
    """Read UTF-8 CSV; preserve order, duplicates and literal strings such as NA."""
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False, na_values=[""])
    validate_news(df)
    return df


def save_processed(df: pd.DataFrame, path: Path) -> None:
    """Write CSV; the caller owns any preceding transformation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def main(input_path: str, output_path: str, sample: int | None = None) -> None:
    """Explicit optional sampling; no hidden deletion or NLP cleaning."""
    df = load_raw_news(Path(input_path))
    if sample is not None:
        if not 1 <= sample <= len(df):
            raise ValueError(f"sample must be between 1 and {len(df)}")
        df = df.sample(n=sample, random_state=42)
    save_processed(df, Path(output_path))
    print(f"Saved dataset: {output_path} (shape={df.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and copy canonical news CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    main(args.input, args.output, args.sample)
