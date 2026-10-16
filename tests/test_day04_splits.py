"""Day04 frozen-input integrity and train-only grouped CV fold checks."""

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.data.experiment_splits import (
    build_train_cv_folds,
    load_day04_inputs,
)
from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.data.load_processed import PROCESSED_COLUMNS
from finnews_sentiment.features.labels_duplicates import LABEL_MAPPING
from finnews_sentiment.features.preprocess import PREPROCESSING_POLICY
from finnews_sentiment.models.run_baseline import run_baseline


def _make_reference(root: Path) -> Path:
    rows = [
        (
            index,
            f"company item{index} outlook {('loss', 'steady', 'growth')[index % 3]}",
            CLASS_ORDER[index % 3],
            f"company item{index} outlook {('loss', 'steady', 'growth')[index % 3]}",
            index % 3,
        )
        for index in range(90)
    ]
    rows.append((90, rows[0][1], rows[0][2], rows[0][3], rows[0][4]))
    frame = pd.DataFrame(rows, columns=PROCESSED_COLUMNS)
    processed_path = root / "data/processed/financial_phrasebank_75agree.csv"
    processed_path.parent.mkdir(parents=True)
    processed_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    processed_path.write_bytes(processed_bytes)

    source = {
        "dataset": "local-test-fixture", "revision": "fixture-revision",
        "url": "https://example.invalid/dataset", "archive_sha256": "a" * 64,
        "subset": "fixture", "member": "Sentences.txt", "encoding": "utf-8",
        "license": "fixture", "citation": "fixture citation", "expected_rows": len(frame),
    }
    dataset_config_path = root / "configs/dataset.json"
    dataset_config_path.parent.mkdir(parents=True)
    dataset_config_bytes = (
        json.dumps({**source, "class_order": list(CLASS_ORDER)}, indent=2) + "\n"
    ).encode("utf-8")
    dataset_config_path.write_bytes(dataset_config_bytes)
    day02_provenance = {
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
            "sha256": hashlib.sha256(processed_bytes).hexdigest(),
            "rows": len(frame), "columns": list(PROCESSED_COLUMNS),
        },
    }
    day02_path = root / "reports/day02/provenance.json"
    day02_path.parent.mkdir(parents=True)
    day02_path.write_text(json.dumps(day02_provenance), encoding="utf-8")

    baseline_config = json.loads((ROOT / "configs/train_baseline.json").read_text(encoding="utf-8"))
    config_path = root / "configs/train_baseline.json"
    config_path.write_text(json.dumps(baseline_config, indent=2) + "\n", encoding="utf-8")
    run_baseline(config_path, root)
    return root


@pytest.fixture(scope="module")
def reference_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_reference(tmp_path_factory.mktemp("day04-reference"))


def _copy_reference(reference_root: Path, tmp_path: Path) -> Path:
    copied = tmp_path / "project"
    shutil.copytree(reference_root, copied)
    return copied


def _update_manifest_checksum(root: Path) -> None:
    provenance_path = root / "reports/day03/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest_path = root / "data/processed/day03/splits.csv"
    provenance["checksums"]["manifest"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")


def test_loads_reference_and_builds_reproducible_train_only_folds(reference_root: Path) -> None:
    before = (reference_root / "data/processed/day03/splits.csv").read_bytes()
    inputs = load_day04_inputs(reference_root)
    train = inputs.splits["train"]
    folds = build_train_cv_folds(train, inputs.train_manifest)
    repeated = build_train_cv_folds(train, inputs.train_manifest)

    assert len(inputs.processed) == len(inputs.manifest) == 91
    assert inputs.manifest["source_row_id"].tolist() == inputs.processed["source_row_id"].tolist()
    assert folds.manifest.equals(repeated.manifest)
    assert folds.summary.equals(repeated.summary)
    assert folds.manifest.columns.tolist() == ["source_row_id", "group_id", "fold_id"]
    assert folds.manifest["source_row_id"].tolist() == train["source_row_id"].tolist()
    assert folds.manifest["fold_id"].value_counts().sort_index().index.tolist() == [0, 1, 2]
    assert folds.manifest.groupby("group_id")["fold_id"].nunique().eq(1).all()
    assert sum(len(fold.validation_positions) for fold in folds.folds) == len(train)
    assert all(set(fold.train_positions).isdisjoint(fold.validation_positions) for fold in folds.folds)
    train_ids = set(train["source_row_id"])
    assert set(folds.manifest["source_row_id"]).issubset(train_ids)
    assert not set(folds.manifest["source_row_id"]).intersection(inputs.splits["validation"]["source_row_id"])
    assert not set(folds.manifest["source_row_id"]).intersection(inputs.splits["test"]["source_row_id"])
    assert (reference_root / "data/processed/day03/splits.csv").read_bytes() == before


def test_manifest_checksum_mismatch_is_rejected(reference_root: Path, tmp_path: Path) -> None:
    root = _copy_reference(reference_root, tmp_path)
    manifest_path = root / "data/processed/day03/splits.csv"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        load_day04_inputs(root)


def test_missing_manifest_gives_reproduction_command(reference_root: Path, tmp_path: Path) -> None:
    root = _copy_reference(reference_root, tmp_path)
    (root / "data/processed/day03/splits.csv").unlink()

    with pytest.raises(FileNotFoundError, match="train_model.*train_baseline.json"):
        load_day04_inputs(root)


def test_wrong_group_id_is_rejected_even_when_checksum_matches(reference_root: Path, tmp_path: Path) -> None:
    root = _copy_reference(reference_root, tmp_path)
    manifest_path = root / "data/processed/day03/splits.csv"
    manifest = pd.read_csv(manifest_path)
    group_ids = manifest["group_id"].to_numpy(dtype="int64", copy=True)
    group_ids[0] += 500
    manifest["group_id"] = group_ids
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    _update_manifest_checksum(root)

    with pytest.raises(ValueError, match=r"group_id must equal min\(source_row_id\)"):
        load_day04_inputs(root)


def test_manifest_must_cover_all_ids_in_canonical_order(reference_root: Path, tmp_path: Path) -> None:
    root = _copy_reference(reference_root, tmp_path)
    manifest_path = root / "data/processed/day03/splits.csv"
    manifest = pd.read_csv(manifest_path).iloc[1:]
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    _update_manifest_checksum(root)

    with pytest.raises(ValueError, match="cover processed source_row_id"):
        load_day04_inputs(root)


def test_duplicate_group_cannot_cross_outer_splits(reference_root: Path, tmp_path: Path) -> None:
    root = _copy_reference(reference_root, tmp_path)
    manifest_path = root / "data/processed/day03/splits.csv"
    manifest = pd.read_csv(manifest_path)
    duplicate_group = int(manifest.loc[manifest["source_row_id"] == 0, "group_id"].iloc[0])
    manifest.loc[manifest["source_row_id"] == 90, "split"] = "validation"
    assert manifest.loc[manifest["group_id"] == duplicate_group, "split"].nunique() > 1
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    _update_manifest_checksum(root)

    with pytest.raises(ValueError, match="text group crosses split boundaries"):
        load_day04_inputs(root)


def test_cv_rejects_outer_validation_rows_and_bad_train_manifest(reference_root: Path) -> None:
    inputs = load_day04_inputs(reference_root)
    with pytest.raises(ValueError, match="only Day03 train rows"):
        build_train_cv_folds(inputs.splits["train"], inputs.manifest.loc[inputs.manifest["split"] != "test"])

    bad_manifest = inputs.train_manifest.copy(deep=True)
    group_ids = bad_manifest["group_id"].to_numpy(dtype="int64", copy=True)
    group_ids[0] += 1
    bad_manifest["group_id"] = group_ids
    with pytest.raises(ValueError, match="group IDs differ"):
        build_train_cv_folds(inputs.splits["train"], bad_manifest)


def test_cv_rejects_train_missing_a_class(reference_root: Path) -> None:
    inputs = load_day04_inputs(reference_root)
    train = inputs.splits["train"]
    neutral_only = train.loc[train["sentiment"] != "negative"].copy()
    train_manifest = inputs.train_manifest.loc[
        inputs.train_manifest["source_row_id"].isin(neutral_only["source_row_id"])
    ].copy()
    with pytest.raises(ValueError, match="All three classes are required"):
        build_train_cv_folds(neutral_only, train_manifest)
