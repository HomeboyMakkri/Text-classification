"""Read-only Day03 input contract, with no split, preprocessing or fitting."""

import csv
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path
import re

import pandas as pd

from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.features.labels_duplicates import (
    LABEL_MAPPING, DuplicateLabelConflictError, analyze_duplicates,
)
from finnews_sentiment.features.preprocess import PREPROCESSING_POLICY

PROCESSED_COLUMNS = ("source_row_id", "text", "sentiment", "text_clean", "label_id")
DAY02_PREPARE_COMMAND = "PYTHONPATH=src venv/bin/python -m finnews_sentiment.data.prepare_day02"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _read_input(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing Day02 artifact: {path}. Prepare it separately from the project root: "
            f"{DAY02_PREPARE_COMMAND}. Source preparation is separate if archive/CSV are absent."
        ) from error


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        result[key] = item
    return result


def _same(value: object, expected: object, name: str) -> None:
    """Strict JSON comparison also rejects bool values masquerading as IDs."""
    if isinstance(expected, dict):
        actual = _object(value, name)
        reference = _object(expected, name)
        if set(actual) != set(reference):
            raise ValueError(f"{name} keys do not match the expected contract")
        for key in reference:
            _same(actual[key], reference[key], f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError(f"{name} does not match {expected!r}")
        for actual, item in zip(value, expected, strict=True):
            _same(actual, item, name)
    elif type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} does not match {expected!r}")


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    return value


def validate_processed_news(df: pd.DataFrame) -> None:
    """Validate a table without modifying its values, row order or index."""
    if tuple(df.columns) != PROCESSED_COLUMNS:
        raise ValueError(f"Processed schema must be exactly {PROCESSED_COLUMNS}")
    if df.empty:
        raise ValueError("Processed dataset contains no rows")
    source_ids: list[int] = []
    raw_to_clean: dict[str, str] = {}
    observed: set[str] = set()
    for source_id, text, sentiment, clean, label in zip(
        df["source_row_id"], df["text"], df["sentiment"], df["text_clean"], df["label_id"], strict=True,
    ):
        if not isinstance(source_id, Integral) or isinstance(source_id, bool) or source_id < 0:
            raise ValueError("source_row_id must contain nonnegative integers")
        source_ids.append(int(source_id))
        if not isinstance(text, str) or not text.strip() or not isinstance(clean, str) or not clean.strip():
            raise ValueError(f"text/text_clean must be nonblank strings; source_row_id={source_id}")
        if not isinstance(sentiment, str) or sentiment not in LABEL_MAPPING:
            raise ValueError(f"Unknown or missing sentiment at source_row_id={source_id}: {sentiment!r}")
        if not isinstance(label, Integral) or isinstance(label, bool) or label != LABEL_MAPPING[sentiment]:
            raise ValueError(f"label_id does not match fixed mapping at source_row_id={source_id}")
        observed.add(sentiment)
        previous = raw_to_clean.setdefault(text, clean)
        if previous != clean:
            raise ValueError("Identical raw text has inconsistent text_clean; check Day02 policy")
    if any(left >= right for left, right in zip(source_ids, source_ids[1:])):
        raise ValueError("source_row_id must be unique and strictly increasing in canonical order")
    if observed != set(CLASS_ORDER):
        raise ValueError(f"All three classes are required; observed={sorted(observed)}")
    duplicates = analyze_duplicates(df)
    if duplicates.has_conflicts:
        raise DuplicateLabelConflictError(duplicates)


def load_processed_news(
    csv_path: Path, provenance_path: Path, dataset_config_path: Path,
) -> pd.DataFrame:
    """Verify Day02 provenance/checksums and read its exact five-column table.

    Read integer fields as strings first to reject float/boolean CSV values
    before explicit int64 conversion. Literal NA/nan texts remain strings.
    Raw/interim files are not reread or rewritten, and no cleanup is applied.
    """
    csv_bytes = _read_input(csv_path)
    provenance_bytes = _read_input(provenance_path)
    config_bytes = dataset_config_path.read_bytes()
    provenance = _object(json.loads(provenance_bytes), "Day02 provenance")
    config = _object(json.loads(config_bytes), "dataset config")
    source_keys = (
        "dataset", "revision", "url", "archive_sha256", "subset", "member", "encoding", "license", "citation",
    )
    expected_source: dict[str, object] = {}
    for key in source_keys:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"dataset config requires nonempty {key}")
        expected_source[key] = value
    expected_rows = config.get("expected_rows")
    if not isinstance(expected_rows, int) or isinstance(expected_rows, bool) or expected_rows < 1:
        raise ValueError("dataset config expected_rows must be a positive integer")
    expected_source["expected_rows"] = expected_rows
    _same(config.get("class_order"), list(CLASS_ORDER), "dataset class_order")
    _same(provenance.get("source"), expected_source, "provenance source")
    _same(provenance.get("dataset_config_sha256"), hashlib.sha256(config_bytes).hexdigest(), "dataset config checksum")
    _same(provenance.get("preprocessing_policy"), PREPROCESSING_POLICY, "preprocessing policy")
    _same(provenance.get("class_order"), list(CLASS_ORDER), "provenance class_order")
    _same(provenance.get("label_mapping"), LABEL_MAPPING, "provenance label_mapping")
    _same(provenance.get("parameters"), {
        "duplicate_policy": "retain_same_label_block_conflicts", "source_row_id": "zero_based_canonical_position",
    }, "provenance parameters")
    archive = _object(provenance.get("archive"), "provenance archive")
    _same(archive.get("path"), "data/raw/FinancialPhraseBank-v1.0.zip", "archive path")
    _same(_hash(archive.get("sha256"), "archive sha256"), config["archive_sha256"], "archive checksum")
    canonical = _object(provenance.get("input"), "provenance input")
    _same(canonical.get("path"), "data/interim/financial_phrasebank_75agree.csv", "canonical path")
    _hash(canonical.get("sha256"), "canonical sha256")
    versions = _object(provenance.get("versions"), "provenance versions")
    for package in ("python", "pandas", "numpy"):
        version = versions.get(package)
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"provenance versions requires {package}")
    output = _object(provenance.get("output"), "provenance output")
    _same(output.get("columns"), list(PROCESSED_COLUMNS), "output columns")
    recorded_path = output.get("path")
    if not isinstance(recorded_path, str) or Path(recorded_path).is_absolute() or ".." in Path(recorded_path).parts:
        raise ValueError("output path must be project-relative")
    root = dataset_config_path.resolve().parent.parent
    if (root / recorded_path).resolve() != csv_path.resolve():
        raise ValueError("CSV path does not match provenance output path")
    _same(_hash(output.get("sha256"), "processed sha256"), hashlib.sha256(csv_bytes).hexdigest(), "processed CSV checksum")
    row_count = output.get("rows")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or not 1 <= row_count <= expected_rows:
        raise ValueError("provenance output rows must be positive and <= source expected_rows")

    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8")), strict=True)
    header = next(reader, [])
    if tuple(header) != PROCESSED_COLUMNS:
        raise ValueError(f"Processed schema must be exactly {PROCESSED_COLUMNS}")
    for line, row in enumerate(reader, start=2):
        if len(row) != len(PROCESSED_COLUMNS):
            raise ValueError(f"CSV record {line} must contain exactly five fields")
    df = pd.read_csv(
        io.BytesIO(csv_bytes), encoding="utf-8", keep_default_na=False, dtype="str", skip_blank_lines=False,
    )
    for column in ("source_row_id", "label_id"):
        if not df[column].str.fullmatch(r"[0-9]+").all():
            raise ValueError(f"{column} CSV values must be nonnegative integer literals")
        try:
            df[column] = df[column].astype("int64")
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{column} CSV values must fit int64") from error
    validate_processed_news(df)
    if len(df) != row_count or int(df["source_row_id"].max()) >= expected_rows:
        raise ValueError("Processed row count/source IDs do not match provenance source/output rows")
    return df
