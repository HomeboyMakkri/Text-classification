"""Explicit preparation of a pinned Financial PhraseBank archive."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from finnews_sentiment.data.load_data import save_processed, validate_news


def parse_phrasebank(content: str) -> pd.DataFrame:
    """Parse sentence@label records, retaining sentence content and source order."""
    records: list[tuple[str, str]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if "@" not in line:
            raise ValueError(f"Missing label separator on line {line_number}")
        text, sentiment = line.rsplit("@", 1)
        if not text.strip():
            raise ValueError(f"Empty sentence on line {line_number}")
        records.append((text, sentiment.strip()))
    df = pd.DataFrame(records, columns=["text", "sentiment"])
    validate_news(df)
    return df


def prepare_phrasebank(root: Path, *, download: bool = False) -> Path:
    """Verify cached bytes and convert one agreement subset to canonical CSV."""
    config = json.loads((root / "configs/dataset.json").read_text(encoding="utf-8"))
    archive_path = root / "data/raw/FinancialPhraseBank-v1.0.zip"
    if archive_path.exists():
        archive_bytes = archive_path.read_bytes()
    elif download:
        with urlopen(config["url"], timeout=60) as response:
            archive_bytes = response.read()
    else:
        raise FileNotFoundError(
            f"Missing {archive_path}; run with --download once to fetch pinned data"
        )
    actual_hash = hashlib.sha256(archive_bytes).hexdigest()
    if actual_hash != config["archive_sha256"]:
        raise ValueError(f"Archive SHA-256 mismatch: {actual_hash}")
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        content = archive.read(config["member"]).decode(config["encoding"])
    df = parse_phrasebank(content)
    if len(df) != config["expected_rows"]:
        raise ValueError(f"Unexpected source row count: {len(df)}")
    if not archive_path.exists():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(archive_bytes)
    output = root / "data/interim/financial_phrasebank_75agree.csv"
    save_processed(df, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--download", action="store_true", help="Fetch only if archive is missing")
    args = parser.parse_args()
    print(prepare_phrasebank(args.root.resolve(), download=args.download))
