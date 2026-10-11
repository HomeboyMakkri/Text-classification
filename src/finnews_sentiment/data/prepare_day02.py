"""Explicit offline preparation of the Day02 CSV and reproducibility reports."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
from zipfile import ZipFile

import pandas as pd

from finnews_sentiment.data.load_data import CLASS_ORDER, load_raw_news
from finnews_sentiment.data.prepare_phrasebank import parse_phrasebank
from finnews_sentiment.features.labels_duplicates import LABEL_MAPPING, prepare_labeled_news
from finnews_sentiment.features.preprocess import PREPROCESSING_POLICY, preprocess_news


@dataclass(frozen=True)
class Day02Artifacts:
    processed_csv: Path
    preprocessing_report: Path
    duplicate_report: Path
    provenance: Path


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def prepare_day02(root: Path) -> Day02Artifacts:
    """Verify local source, run shared preprocessing and save deterministic outputs.

    No download or training. All source/contract/conflict checks finish before
    output files are written; raw and interim are only read.
    """
    root = root.resolve()
    config_path = root / "configs/dataset.json"
    config_bytes = config_path.read_bytes()
    payload: object = json.loads(config_bytes)
    if not isinstance(payload, dict):
        raise ValueError("dataset.json must contain an object")
    source: dict[str, str] = {}
    for key in ("dataset", "revision", "url", "archive_sha256", "subset", "member", "encoding", "license", "citation"):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"dataset.json requires nonempty string {key!r}")
        source[key] = value
    expected_rows = payload.get("expected_rows")
    if not isinstance(expected_rows, int) or isinstance(expected_rows, bool) or expected_rows < 1:
        raise ValueError("expected_rows must be a positive integer")
    if payload.get("class_order") != list(CLASS_ORDER):
        raise ValueError("Source class_order does not match project contract")

    archive_path = root / "data/raw/FinancialPhraseBank-v1.0.zip"
    canonical_path = root / "data/interim/financial_phrasebank_75agree.csv"
    archive_bytes = archive_path.read_bytes()
    canonical_bytes = canonical_path.read_bytes()
    archive_hash = _sha256(archive_bytes)
    if archive_hash != source["archive_sha256"]:
        raise ValueError("Archive SHA-256 mismatch; Day02 preparation aborted")
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        original = parse_phrasebank(archive.read(source["member"]).decode(source["encoding"]))
    raw = load_raw_news(canonical_path)
    if len(raw) != expected_rows:
        raise ValueError(f"Unexpected canonical row count: {len(raw)}; expected {expected_rows}")
    try:
        pd.testing.assert_frame_equal(raw, original)
    except AssertionError as error:
        raise ValueError("Canonical CSV differs from pinned archive conversion") from error

    processed, preprocessing_report = preprocess_news(raw)
    labeled, duplicates = prepare_labeled_news(processed)
    csv_bytes = labeled.to_csv(index=False, lineterminator="\n").encode("utf-8")
    # Explicit dtypes preserve literal NA/nan and numeric-only text on CSV reload.
    reloaded = pd.read_csv(
        io.BytesIO(csv_bytes), encoding="utf-8", keep_default_na=False,
        dtype={"source_row_id": "int64", "text": "str", "sentiment": "str", "text_clean": "str", "label_id": "int64"},
    )
    pd.testing.assert_frame_equal(labeled, reloaded)
    duplicate_payload = {
        **asdict(duplicates),
        "raw_duplicate_rows": duplicates.raw_duplicate_rows,
        "clean_duplicate_rows": duplicates.clean_duplicate_rows,
        "new_clean_groups": sum(group.new_after_cleaning for group in duplicates.clean_groups),
        "raw_conflicts": sum(group.conflicting for group in duplicates.raw_groups),
        "clean_conflicts": sum(group.conflicting for group in duplicates.clean_groups),
    }
    output_relative_path = "data/processed/financial_phrasebank_75agree.csv"
    provenance = {
        "source": {**source, "expected_rows": expected_rows},
        "input": {"path": str(canonical_path.relative_to(root)), "sha256": _sha256(canonical_bytes)},
        "archive": {"path": str(archive_path.relative_to(root)), "sha256": archive_hash},
        "dataset_config_sha256": _sha256(config_bytes),
        "preprocessing_policy": PREPROCESSING_POLICY,
        "parameters": {"duplicate_policy": "retain_same_label_block_conflicts", "source_row_id": "zero_based_canonical_position"},
        "class_order": list(CLASS_ORDER), "label_mapping": LABEL_MAPPING,
        "versions": {"python": platform.python_version(), "pandas": version("pandas"), "numpy": version("numpy")},
        "output": {
            "path": output_relative_path,
            "sha256": _sha256(csv_bytes), "rows": len(labeled), "columns": list(labeled.columns),
        },
    }
    artifacts = Day02Artifacts(
        processed_csv=root / output_relative_path,
        preprocessing_report=root / "reports/day02/preprocessing.json",
        duplicate_report=root / "reports/day02/duplicates.json",
        provenance=root / "reports/day02/provenance.json",
    )
    outputs = (
        (artifacts.processed_csv, csv_bytes),
        (artifacts.preprocessing_report, _json_bytes(asdict(preprocessing_report))),
        (artifacts.duplicate_report, _json_bytes(duplicate_payload)),
        (artifacts.provenance, _json_bytes(provenance)),
    )
    if canonical_path.read_bytes() != canonical_bytes or archive_path.read_bytes() != archive_bytes:
        raise ValueError("Source changed during preparation; outputs not written")
    for path, content in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    artifacts = prepare_day02(args.root)
    for name, path in asdict(artifacts).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
