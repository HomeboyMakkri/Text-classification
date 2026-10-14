"""Offline reference artifact, provenance and determinism checks."""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.data.load_processed import PROCESSED_COLUMNS
from finnews_sentiment.features.labels_duplicates import LABEL_MAPPING
from finnews_sentiment.features.preprocess import PREPROCESSING_POLICY
from finnews_sentiment.models.run_baseline import run_baseline


def _fixture_project(root: Path) -> Path:
    rows = [
        (i, f"company item{i} outlook {('loss', 'steady', 'growth')[i % 3]}",
         CLASS_ORDER[i % 3], f"company item{i} outlook {('loss', 'steady', 'growth')[i % 3]}", i % 3)
        for i in range(90)
    ]
    rows.append((90, rows[0][1], rows[0][2], rows[0][3], rows[0][4]))
    frame = pd.DataFrame(rows, columns=PROCESSED_COLUMNS)
    csv_path = root / "data/processed/financial_phrasebank_75agree.csv"
    csv_path.parent.mkdir(parents=True)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    csv_path.write_bytes(csv_bytes)

    source = {
        "dataset": "local-test-fixture", "revision": "fixture-revision",
        "url": "https://example.invalid/dataset", "archive_sha256": "a" * 64,
        "subset": "fixture", "member": "Sentences.txt", "encoding": "utf-8",
        "license": "fixture", "citation": "fixture citation", "expected_rows": 91,
    }
    dataset_config = {**source, "class_order": list(CLASS_ORDER)}
    dataset_config_path = root / "configs/dataset.json"
    dataset_config_path.parent.mkdir(parents=True)
    dataset_config_bytes = (json.dumps(dataset_config, indent=2) + "\n").encode("utf-8")
    dataset_config_path.write_bytes(dataset_config_bytes)
    provenance = {
        "source": source,
        "dataset_config_sha256": hashlib.sha256(dataset_config_bytes).hexdigest(),
        "input": {"path": "data/interim/financial_phrasebank_75agree.csv", "sha256": "b" * 64},
        "archive": {"path": "data/raw/FinancialPhraseBank-v1.0.zip", "sha256": source["archive_sha256"]},
        "preprocessing_policy": PREPROCESSING_POLICY,
        "parameters": {
            "duplicate_policy": "retain_same_label_block_conflicts",
            "source_row_id": "zero_based_canonical_position",
        },
        "class_order": list(CLASS_ORDER), "label_mapping": LABEL_MAPPING,
        "versions": {"python": "fixture", "pandas": "fixture", "numpy": "fixture"},
        "output": {
            "path": "data/processed/financial_phrasebank_75agree.csv",
            "sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "rows": len(frame), "columns": list(PROCESSED_COLUMNS),
        },
    }
    provenance_path = root / "reports/day02/provenance.json"
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    config = json.loads((ROOT / "configs/train_baseline.json").read_text(encoding="utf-8"))
    config["input_path"] = "data/processed/financial_phrasebank_75agree.csv"
    config["provenance_path"] = "reports/day02/provenance.json"
    config["split_output_dir"] = "data/processed/day03"
    config["report_output_dir"] = "reports/day03"
    config_path = root / "configs/train_baseline.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def test_runner_roundtrips_artifacts_and_reproduces_bytes(tmp_path: Path) -> None:
    config_path = _fixture_project(tmp_path)

    first = run_baseline(config_path, tmp_path)
    paths = (
        first.artifacts.splits, first.artifacts.split_summary,
        first.artifacts.validation_predictions, first.artifacts.metrics,
        first.artifacts.config, first.artifacts.provenance,
    )
    first_bytes = {path: path.read_bytes() for path in paths}
    second = run_baseline(config_path, tmp_path)

    assert {path: path.read_bytes() for path in paths} == first_bytes
    assert second.fit.validation_predictions.equals(first.fit.validation_predictions)
    manifest = pd.read_csv(first.artifacts.splits, dtype={"source_row_id": "int64", "group_id": "int64"})
    predictions = pd.read_csv(first.artifacts.validation_predictions, dtype="int64")
    metrics = json.loads(first.artifacts.metrics.read_text(encoding="utf-8"))
    provenance = json.loads(first.artifacts.provenance.read_text(encoding="utf-8"))

    assert manifest.columns.tolist() == ["source_row_id", "group_id", "split"]
    validation_ids = manifest.loc[manifest["split"] == "validation", "source_row_id"].tolist()
    assert predictions["source_row_id"].tolist() == validation_ids
    assert predictions.columns.tolist() == [
        "source_row_id", "true_label_id", "baseline_pred_label_id", "dummy_pred_label_id",
    ]
    assert metrics["evaluation_split"] == "validation"
    assert metrics["n_valid"] == len(validation_ids)
    assert metrics["class_order"] == list(CLASS_ORDER)
    assert sum(item["support"] for item in metrics["baseline"]["per_class"]) == len(validation_ids)
    assert provenance["source"] == {
        "dataset": "local-test-fixture", "revision": "fixture-revision", "subset": "fixture",
    }
    assert provenance["checksums"]["manifest"] == hashlib.sha256(first_bytes[first.artifacts.splits]).hexdigest()
    assert provenance["checksums"]["predictions"] == hashlib.sha256(first_bytes[first.artifacts.validation_predictions]).hexdigest()
    assert provenance["checksums"]["metrics"] == hashlib.sha256(first_bytes[first.artifacts.metrics]).hexdigest()
    assert "timestamp" not in provenance


def test_runner_writes_all_declared_outputs(tmp_path: Path) -> None:
    config_path = _fixture_project(tmp_path)
    result = run_baseline(config_path, tmp_path)

    assert all(path.is_file() for path in (
        result.artifacts.splits, result.artifacts.split_summary,
        result.artifacts.validation_predictions, result.artifacts.metrics,
        result.artifacts.config, result.artifacts.provenance,
    ))


def test_runner_requires_local_verified_day02_artifacts(tmp_path: Path) -> None:
    config_path = _fixture_project(tmp_path)
    (tmp_path / "data/processed/financial_phrasebank_75agree.csv").unlink()

    with pytest.raises(FileNotFoundError, match="prepare_day02"):
        run_baseline(config_path, tmp_path)
    assert not (tmp_path / "reports/day03").exists()
