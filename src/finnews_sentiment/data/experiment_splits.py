"""Load the fixed Day03 split and create train-only grouped CV folds."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from finnews_sentiment.data.baseline_config import load_baseline_config
from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.data.load_processed import load_processed_news, validate_processed_news
from finnews_sentiment.features.labels_duplicates import LABEL_MAPPING

DAY03_SPLIT_ORDER = ("train", "validation", "test")
DAY03_MANIFEST_COLUMNS = ("source_row_id", "group_id", "split")
CV_MANIFEST_COLUMNS = ("source_row_id", "group_id", "fold_id")
DAY03_REBUILD_COMMAND = (
    "PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.train_model "
    "--config configs/train_baseline.json"
)
_INTEGER = re.compile(r"[0-9]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Day04Inputs:
    """Verified processed rows reconstructed using the frozen Day03 manifest."""

    processed: pd.DataFrame
    manifest: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    train_manifest: pd.DataFrame
    split_summary: pd.DataFrame
    validation_predictions: pd.DataFrame
    metrics: dict[str, object]
    provenance: dict[str, object]


@dataclass(frozen=True)
class CVFold:
    """Positional train/validation indexes into the ordered Day03 train rows."""

    fold_id: int
    train_positions: tuple[int, ...]
    validation_positions: tuple[int, ...]


@dataclass(frozen=True)
class CVFoldSet:
    """One fixed fold assignment and the positional folds consumed by later IDs."""

    manifest: pd.DataFrame
    folds: tuple[CVFold, ...]
    summary: pd.DataFrame
    n_splits: int
    random_state: int


def _read_object(path: Path, name: str) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing Day03 artifact {path}. Rebuild the frozen reference from the project root: "
            f"{DAY03_REBUILD_COMMAND}"
        ) from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return payload


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


def _required_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"Day03 provenance {name} must be a lowercase SHA-256 digest")
    return value


def _verify_hash(path: Path, expected: object, name: str) -> bytes:
    try:
        content = path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing Day03 artifact {path}. Rebuild the frozen reference from the project root: "
            f"{DAY03_REBUILD_COMMAND}"
        ) from error
    expected_hash = _required_hash(expected, f"checksums.{name}")
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"Day03 {name} checksum mismatch: {path}")
    return content


def _read_manifest(content: bytes) -> pd.DataFrame:
    frame = pd.read_csv(
        BytesIO(content), encoding="utf-8", keep_default_na=False, dtype="str",
    )
    if tuple(frame.columns) != DAY03_MANIFEST_COLUMNS:
        raise ValueError(f"Day03 split manifest columns must be {DAY03_MANIFEST_COLUMNS}")
    for column in ("source_row_id", "group_id"):
        if not frame[column].str.fullmatch(_INTEGER).all():
            raise ValueError(f"Day03 manifest {column} values must be nonnegative integer literals")
        frame[column] = frame[column].astype("int64")
    return frame


def _validate_effective_config(
    effective: dict[str, object], config_path: Path, config_hash: str,
) -> None:
    config = load_baseline_config(config_path)
    checks: dict[str, object] = {
        "policy": config.policy,
        "input_path": str(config.input_path),
        "provenance_path": str(config.provenance_path),
        "text_col": config.text_col,
        "target_col": config.target_col,
        "random_state": config.random_state,
        "test_size": config.test_size,
        "validation_size": config.validation_size,
        "tfidf": {**config.tfidf, "ngram_range": list(config.tfidf["ngram_range"])},
        "logreg": dict(config.logreg),
        "dummy": dict(config.dummy),
        "split_output_dir": str(config.split_output_dir),
        "report_output_dir": str(config.report_output_dir),
        "split_algorithm": "group-stratified-v1",
    }
    if effective != checks:
        raise ValueError("Day03 effective config differs from configs/train_baseline.json")
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != config_hash:
        raise ValueError("Day03 baseline config checksum mismatch")


def _validate_split_summary(
    summary: dict[str, object], manifest: pd.DataFrame, splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if summary.get("algorithm") != "group-stratified-v1" or summary.get("random_state") != 42:
        raise ValueError("Day03 split summary algorithm/seed differs from the frozen contract")
    if summary.get("rows") != len(manifest) or summary.get("groups") != int(manifest["group_id"].nunique()):
        raise ValueError("Day03 split summary row/group totals differ from the manifest")
    raw_splits = _object(summary.get("splits"), "Day03 summary splits")
    rows: list[dict[str, object]] = []
    for split_name in DAY03_SPLIT_ORDER:
        stats = _object(raw_splits.get(split_name), f"Day03 summary {split_name}")
        frame = splits[split_name]
        group_count = int(frame["text_clean"].nunique())
        row_counts = {
            label: int(frame["sentiment"].eq(label).sum()) for label in CLASS_ORDER
        }
        unique = frame.drop_duplicates(subset=["text_clean"])
        group_counts = {
            label: int(unique["sentiment"].eq(label).sum()) for label in CLASS_ORDER
        }
        if stats.get("rows") != len(frame) or stats.get("groups") != group_count:
            raise ValueError(f"Day03 {split_name} summary sizes differ from restored rows")
        if stats.get("row_class_counts") != row_counts or stats.get("group_class_counts") != group_counts:
            raise ValueError(f"Day03 {split_name} summary class counts differ from restored rows")
        rows.append({"split": split_name, "rows": len(frame), "groups": group_count, **row_counts})
    disjoint = _object(summary.get("disjoint"), "Day03 summary disjoint")
    if disjoint != {"source_row_id": True, "text": True, "text_clean": True}:
        raise ValueError("Day03 split summary does not claim all required disjointness checks")
    return pd.DataFrame(rows).set_index("split")


def load_day04_inputs(root: Path) -> Day04Inputs:
    """Verify and restore the exact Day03 rows without recomputing its split.

    Day02's loader verifies source data and provenance. Day03 checksums protect
    the frozen manifest, summary, predictions, metrics and effective config.
    The rows are then reassembled by source_row_id in canonical input order.
    """
    project_root = root.resolve()
    config_path = project_root / "configs/train_baseline.json"
    config = load_baseline_config(config_path)
    report_dir = project_root / config.report_output_dir
    provenance = _read_object(report_dir / "provenance.json", "Day03 provenance")
    checksums = _object(provenance.get("checksums"), "Day03 checksums")
    if provenance.get("random_state") != config.random_state or provenance.get("split_algorithm") != "group-stratified-v1":
        raise ValueError("Day03 provenance split algorithm/seed differs from the frozen config")
    classes = _object(provenance.get("classes"), "Day03 classes")
    if classes.get("class_order") != list(CLASS_ORDER) or classes.get("label_mapping") != LABEL_MAPPING:
        raise ValueError("Day03 provenance class order/mapping differs from the fixed sentiment contract")
    if classes.get("logreg_classes") != [0, 1, 2] or classes.get("dummy_classes") != [0, 1, 2]:
        raise ValueError("Day03 provenance estimator classes differ from the fixed three-class order")
    versions = _object(provenance.get("versions"), "Day03 versions")
    sklearn_version = versions.get("scikit-learn")
    if not isinstance(sklearn_version, str) or not sklearn_version.strip():
        raise ValueError("Day03 provenance must record the scikit-learn version")

    manifest_path = project_root / config.split_output_dir / "splits.csv"
    summary_path = report_dir / "split_summary.json"
    predictions_path = report_dir / "validation_predictions.csv"
    metrics_path = report_dir / "metrics.json"
    effective_config_path = report_dir / "config.json"
    manifest_bytes = _verify_hash(manifest_path, checksums.get("manifest"), "manifest")
    summary_bytes = _verify_hash(summary_path, checksums.get("split_summary"), "split_summary")
    predictions_bytes = _verify_hash(predictions_path, checksums.get("predictions"), "predictions")
    metrics_bytes = _verify_hash(metrics_path, checksums.get("metrics"), "metrics")
    effective_config_bytes = _verify_hash(
        effective_config_path, checksums.get("effective_config"), "effective_config",
    )
    if hashlib.sha256((project_root / config.input_path).read_bytes()).hexdigest() != checksums.get("processed_csv"):
        raise ValueError("Day03 processed CSV checksum differs from provenance")
    config_hash = _required_hash(checksums.get("config"), "checksums.config")
    effective_config = _object(json.loads(effective_config_bytes), "Day03 effective config")
    _validate_effective_config(effective_config, config_path, config_hash)

    processed = load_processed_news(
        project_root / config.input_path,
        project_root / config.provenance_path,
        project_root / "configs/dataset.json",
    )

    input_checksums = _object(provenance.get("input_checksums"), "Day03 input checksums")
    required_inputs = {
        "configs/train_baseline.json": config_path,
        "configs/dataset.json": project_root / "configs/dataset.json",
        str(config.input_path): project_root / config.input_path,
        str(config.provenance_path): project_root / config.provenance_path,
    }
    for relative_path, input_path in required_inputs.items():
        expected_hash = _required_hash(input_checksums.get(relative_path), f"input_checksums.{relative_path}")
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Day03 input checksum mismatch: {relative_path}")
    day02_hash = _required_hash(provenance.get("day02_provenance_sha256"), "day02_provenance_sha256")
    if hashlib.sha256((project_root / config.provenance_path).read_bytes()).hexdigest() != day02_hash:
        raise ValueError("Day03 Day02 provenance checksum mismatch")
    manifest = _read_manifest(manifest_bytes)
    expected_ids = processed["source_row_id"].tolist()
    if manifest["source_row_id"].tolist() != expected_ids:
        raise ValueError("Day03 manifest must cover processed source_row_id values exactly in canonical order")
    if manifest["source_row_id"].duplicated().any():
        raise ValueError("Day03 manifest source_row_id values must be unique")
    if not manifest["split"].isin(DAY03_SPLIT_ORDER).all():
        raise ValueError(f"Day03 manifest split values must be one of {DAY03_SPLIT_ORDER}")
    if set(manifest["split"]) != set(DAY03_SPLIT_ORDER):
        raise ValueError("Day03 manifest must contain train, validation and test rows")

    clean_to_group = (
        processed.groupby("text_clean", sort=False)["source_row_id"].min().astype("int64").to_dict()
    )
    expected_group_ids = processed["text_clean"].map(clean_to_group).tolist()
    if manifest["group_id"].tolist() != expected_group_ids:
        raise ValueError("Day03 manifest group_id must equal min(source_row_id) per exact text_clean group")
    group_split_counts = manifest.groupby("group_id", sort=False)["split"].nunique()
    if not group_split_counts.eq(1).all():
        raise ValueError("A Day03 text group crosses split boundaries")

    split_by_id = dict(zip(manifest["source_row_id"], manifest["split"], strict=True))
    restored = processed.copy(deep=True)
    restored["split"] = restored["source_row_id"].map(split_by_id)
    splits = {
        split_name: restored.loc[restored["split"] == split_name].drop(columns="split").copy(deep=True)
        for split_name in DAY03_SPLIT_ORDER
    }
    for split_name, frame in splits.items():
        if set(frame["sentiment"]) != set(CLASS_ORDER):
            raise ValueError(f"All three classes are required in restored Day03 {split_name}")
    for index, left_name in enumerate(DAY03_SPLIT_ORDER):
        for right_name in DAY03_SPLIT_ORDER[index + 1:]:
            left, right = splits[left_name], splits[right_name]
            for column in ("source_row_id", "text", "text_clean"):
                if not set(left[column]).isdisjoint(right[column]):
                    raise ValueError(f"Day03 {column} values overlap between {left_name} and {right_name}")

    summary = _object(json.loads(summary_bytes), "Day03 split summary")
    split_summary = _validate_split_summary(summary, manifest, splits)
    predictions = pd.read_csv(
        BytesIO(predictions_bytes), encoding="utf-8", keep_default_na=False,
        dtype="str",
    )
    expected_prediction_columns = (
        "source_row_id", "true_label_id", "baseline_pred_label_id", "dummy_pred_label_id",
    )
    if tuple(predictions.columns) != expected_prediction_columns:
        raise ValueError(f"Day03 validation prediction columns must be {expected_prediction_columns}")
    for column in expected_prediction_columns:
        if not predictions[column].str.fullmatch(_INTEGER).all():
            raise ValueError(f"Day03 prediction {column} values must be nonnegative integer literals")
        predictions[column] = predictions[column].astype("int64")
    validation = splits["validation"]
    if predictions["source_row_id"].tolist() != validation["source_row_id"].tolist():
        raise ValueError("Day03 validation predictions are not aligned with frozen validation source IDs")
    expected_labels = validation["label_id"].tolist()
    if predictions["true_label_id"].tolist() != expected_labels:
        raise ValueError("Day03 validation prediction truth labels differ from processed validation rows")
    for column in expected_prediction_columns[1:]:
        if not predictions[column].isin((0, 1, 2)).all():
            raise ValueError(f"Day03 prediction {column} contains unknown class IDs")

    metrics = _object(json.loads(metrics_bytes), "Day03 metrics")
    if (
        metrics.get("evaluation_split") != "validation"
        or metrics.get("class_order") != list(CLASS_ORDER)
        or metrics.get("label_mapping") != LABEL_MAPPING
    ):
        raise ValueError("Day03 metrics must describe validation and preserve the fixed class order")
    if metrics.get("n_valid") != len(validation):
        raise ValueError("Day03 validation metric denominator differs from the frozen validation rows")
    train_manifest = manifest.loc[manifest["split"] == "train"].copy(deep=True).reset_index(drop=True)
    return Day04Inputs(
        processed=processed, manifest=manifest, splits=splits, train_manifest=train_manifest,
        split_summary=split_summary, validation_predictions=predictions,
        metrics=metrics, provenance=provenance,
    )


def build_train_cv_folds(
    train: pd.DataFrame,
    train_manifest: pd.DataFrame,
    *,
    n_splits: int = 3,
    random_state: int = 42,
) -> CVFoldSet:
    """Create deterministic, stratified, group-isolated folds on Day03 train only.

    ``train_manifest`` must be the corresponding Day03 train rows, in the same
    order, carrying frozen source/group IDs. The returned manifest assigns each
    train row to its single CV validation fold; fold training rows are its complement.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    if isinstance(random_state, bool) or not isinstance(random_state, int) or not 0 <= random_state < 2**32:
        raise ValueError("random_state must be an integer between 0 and 2**32 - 1")
    if tuple(train.columns) != ("source_row_id", "text", "sentiment", "text_clean", "label_id"):
        raise ValueError("CV train must have the exact processed Day03 schema")
    validate_processed_news(train)
    if tuple(train_manifest.columns) != DAY03_MANIFEST_COLUMNS:
        raise ValueError(f"CV train manifest columns must be {DAY03_MANIFEST_COLUMNS}")
    if train_manifest["split"].tolist() != ["train"] * len(train_manifest):
        raise ValueError("CV manifest input may contain only Day03 train rows")
    if train["source_row_id"].tolist() != train_manifest["source_row_id"].tolist():
        raise ValueError("CV train rows and Day03 train manifest IDs/order do not match")
    clean_to_group = train.groupby("text_clean", sort=False)["source_row_id"].min().astype("int64").to_dict()
    expected_groups = train["text_clean"].map(clean_to_group).tolist()
    if train_manifest["group_id"].tolist() != expected_groups:
        raise ValueError("CV group IDs differ from the frozen exact-text group assignment")
    if set(train["sentiment"]) != set(CLASS_ORDER):
        raise ValueError("All three sentiment classes are required in Day03 train")
    if train_manifest["group_id"].nunique() < n_splits:
        raise ValueError("Day03 train has fewer text groups than requested CV folds")

    y = train["label_id"].to_numpy()
    groups = train_manifest["group_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[CVFold] = []
    assignment = [-1] * len(train)
    summary_rows: list[dict[str, object]] = []
    try:
        split_indices = splitter.split(train["text_clean"], y, groups)
        for fold_id, (train_positions, validation_positions) in enumerate(split_indices):
            if set(groups[train_positions]).intersection(groups[validation_positions]):
                raise ValueError(f"Text groups overlap between CV train and validation in fold {fold_id}")
            train_classes = set(y[train_positions].tolist())
            validation_classes = set(y[validation_positions].tolist())
            if train_classes != {0, 1, 2} or validation_classes != {0, 1, 2}:
                raise ValueError(
                    f"All classes are required in CV fold {fold_id}; "
                    f"train={sorted(train_classes)}, validation={sorted(validation_classes)}"
                )
            if any(assignment[position] != -1 for position in validation_positions):
                raise ValueError("CV validation folds overlap in source rows")
            for position in validation_positions:
                assignment[int(position)] = fold_id
            folds.append(CVFold(
                fold_id=fold_id,
                train_positions=tuple(int(value) for value in train_positions),
                validation_positions=tuple(int(value) for value in validation_positions),
            ))
            valid_labels = y[validation_positions]
            summary_rows.append({
                "fold_id": fold_id,
                "train_rows": len(train_positions),
                "train_groups": int(pd.Series(groups[train_positions]).nunique()),
                "validation_rows": len(validation_positions),
                "validation_groups": int(pd.Series(groups[validation_positions]).nunique()),
                **{
                    f"train_{label}_rows": int((y[train_positions] == class_id).sum())
                    for class_id, label in enumerate(CLASS_ORDER)
                },
                **{
                    f"validation_{label}_rows": int((valid_labels == class_id).sum())
                    for class_id, label in enumerate(CLASS_ORDER)
                },
            })
    except ValueError as error:
        if "CV fold" in str(error) or "CV validation" in str(error) or "Text groups overlap" in str(error):
            raise
        raise ValueError(f"Cannot create {n_splits} StratifiedGroupKFold folds: {error}") from error

    if len(folds) != n_splits or any(fold_id < 0 for fold_id in assignment):
        raise ValueError("CV validation folds must cover every Day03 train row exactly once")
    fold_manifest = train_manifest.loc[:, ["source_row_id", "group_id"]].copy(deep=True)
    fold_manifest["fold_id"] = assignment
    if fold_manifest.groupby("group_id", sort=False)["fold_id"].nunique().ne(1).any():
        raise ValueError("A duplicate text group was assigned to multiple CV validation folds")
    return CVFoldSet(
        manifest=fold_manifest.loc[:, list(CV_MANIFEST_COLUMNS)].copy(deep=True),
        folds=tuple(folds), summary=pd.DataFrame(summary_rows).set_index("fold_id"),
        n_splits=n_splits, random_state=random_state,
    )
