"""Run the complete offline Day03 baseline and write reproducible references."""

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
from typing import cast

import numpy as np
import pandas as pd

from finnews_sentiment.data.baseline_config import BaselineConfig, load_baseline_config
from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.data.load_processed import load_processed_news
from finnews_sentiment.data.split_data import split_processed_news
from finnews_sentiment.features.labels_duplicates import LABEL_MAPPING
from finnews_sentiment.features.preprocess import PREPROCESSING_POLICY
from finnews_sentiment.models.evaluate import BaselineEvaluation, evaluate_baselines
from finnews_sentiment.models.train_model import BaselineFit, fit_baselines


@dataclass(frozen=True)
class Day03ArtifactPaths:
    splits: Path
    split_summary: Path
    validation_predictions: Path
    metrics: Path
    config: Path
    provenance: Path


@dataclass(frozen=True)
class BaselineRun:
    """In-memory run results and the paths of written reference artifacts."""

    config: BaselineConfig
    fit: BaselineFit
    evaluation: BaselineEvaluation
    artifacts: Day03ArtifactPaths


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _effective_config(config: BaselineConfig, split_algorithm: str) -> dict[str, object]:
    return {
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
        "split_algorithm": split_algorithm,
    }


def _metrics_payload(evaluation: BaselineEvaluation) -> dict[str, object]:
    baseline = asdict(evaluation.baseline)
    baseline["per_class"] = [asdict(item) for item in evaluation.baseline.per_class]
    dummy = asdict(evaluation.dummy)
    dummy["per_class"] = [asdict(item) for item in evaluation.dummy.per_class]
    return {
        "evaluation_split": "validation",
        "n_valid": evaluation.baseline.n_samples,
        "class_order": list(CLASS_ORDER),
        "label_mapping": LABEL_MAPPING,
        "baseline": baseline,
        "dummy": dummy,
    }


def _artifact_paths(root: Path, config: BaselineConfig) -> Day03ArtifactPaths:
    return Day03ArtifactPaths(
        splits=root / config.split_output_dir / "splits.csv",
        split_summary=root / config.report_output_dir / "split_summary.json",
        validation_predictions=root / config.report_output_dir / "validation_predictions.csv",
        metrics=root / config.report_output_dir / "metrics.json",
        config=root / config.report_output_dir / "config.json",
        provenance=root / config.report_output_dir / "provenance.json",
    )


def _input_paths(root: Path, config_path: Path, config: BaselineConfig) -> tuple[Path, ...]:
    paths = (
        config_path,
        root / "configs/dataset.json",
        root / config.input_path,
        root / config.provenance_path,
        root / "data/raw/FinancialPhraseBank-v1.0.zip",
        root / "data/interim/financial_phrasebank_75agree.csv",
    )
    return tuple(path for path in paths if path.is_file())


def _roundtrip_csv(content: bytes, *, ids: tuple[str, ...], columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(content), encoding="utf-8", keep_default_na=False,
                            dtype={column: "int64" for column in ids})
    except (ValueError, pd.errors.ParserError) as error:
        raise ValueError("Generated Day03 CSV failed its read-back check") from error
    if tuple(frame.columns) != columns:
        raise ValueError(f"Generated Day03 CSV columns differ from {columns}")
    return frame


def run_baseline(config_path: Path, root: Path | None = None) -> BaselineRun:
    """Load, split, train, evaluate and save deterministic Day03 artifacts offline."""
    project_root = (root or Path(__file__).resolve().parents[3]).resolve()
    resolved_config_path = config_path if config_path.is_absolute() else project_root / config_path
    resolved_config_path = resolved_config_path.resolve()
    config = load_baseline_config(resolved_config_path)
    input_paths = _input_paths(project_root, resolved_config_path, config)
    input_bytes = {path: path.read_bytes() for path in input_paths}

    dataset_config_path = project_root / "configs/dataset.json"
    processed = load_processed_news(
        project_root / config.input_path,
        project_root / config.provenance_path,
        dataset_config_path,
    )
    splits = split_processed_news(
        processed,
        test_size=config.test_size,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )
    fit = fit_baselines(splits, config)
    validation = splits.splits["validation"]
    predictions = fit.validation_predictions
    expected_ids = validation["source_row_id"].tolist()
    if predictions["source_row_id"].tolist() != expected_ids:
        raise ValueError("Validation predictions are not aligned to validation source IDs")
    prediction_frame = pd.DataFrame({
        "source_row_id": validation["source_row_id"].to_numpy(copy=True),
        "true_label_id": validation[config.target_col].to_numpy(copy=True),
        "baseline_pred_label_id": predictions["baseline_prediction"].to_numpy(copy=True),
        "dummy_pred_label_id": predictions["dummy_prediction"].to_numpy(copy=True),
    })
    evaluation = evaluate_baselines(
        prediction_frame["true_label_id"].to_numpy(dtype=np.int64),
        prediction_frame["baseline_pred_label_id"].to_numpy(dtype=np.int64),
        prediction_frame["dummy_pred_label_id"].to_numpy(dtype=np.int64),
    )
    artifacts = _artifact_paths(project_root, config)

    manifest_bytes = splits.manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    split_summary_bytes = _json_bytes(asdict(splits.summary))
    predictions_bytes = prediction_frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    metrics_bytes = _json_bytes(_metrics_payload(evaluation))
    effective_config = _effective_config(config, splits.summary.algorithm)
    config_bytes = _json_bytes(effective_config)

    manifest_roundtrip = _roundtrip_csv(
        manifest_bytes, ids=("source_row_id", "group_id"),
        columns=("source_row_id", "group_id", "split"),
    )
    pd.testing.assert_frame_equal(manifest_roundtrip, splits.manifest, check_dtype=True)
    predictions_roundtrip = _roundtrip_csv(
        predictions_bytes,
        ids=("source_row_id", "true_label_id", "baseline_pred_label_id", "dummy_pred_label_id"),
        columns=("source_row_id", "true_label_id", "baseline_pred_label_id", "dummy_pred_label_id"),
    )
    pd.testing.assert_frame_equal(predictions_roundtrip, prediction_frame, check_dtype=True)
    roundtrip_evaluation = evaluate_baselines(
        predictions_roundtrip["true_label_id"].to_numpy(dtype=np.int64),
        predictions_roundtrip["baseline_pred_label_id"].to_numpy(dtype=np.int64),
        predictions_roundtrip["dummy_pred_label_id"].to_numpy(dtype=np.int64),
    )
    if _metrics_payload(roundtrip_evaluation) != _metrics_payload(evaluation):
        raise ValueError("Round-tripped validation predictions do not reproduce metrics")
    if json.loads(split_summary_bytes) != asdict(splits.summary):
        raise ValueError("Generated split summary failed its JSON round-trip check")
    if json.loads(metrics_bytes) != _metrics_payload(evaluation):
        raise ValueError("Generated metrics failed their JSON round-trip check")
    if json.loads(config_bytes) != effective_config:
        raise ValueError("Generated effective config failed its JSON round-trip check")
    if predictions_roundtrip["source_row_id"].tolist() != expected_ids:
        raise ValueError("Prediction CSV row order differs from validation order")
    if len(predictions_roundtrip) != evaluation.baseline.n_samples or evaluation.baseline.n_samples != evaluation.dummy.n_samples:
        raise ValueError("Predictions and metric denominators are inconsistent")

    day02_provenance = json.loads((project_root / config.provenance_path).read_bytes())
    if not isinstance(day02_provenance, dict):
        raise ValueError("Day02 provenance must be a JSON object")
    source = day02_provenance.get("source")
    if not isinstance(source, dict):
        raise ValueError("Day02 provenance is missing source metadata")
    versions = {
        "python": platform.python_version(),
        "pandas": version("pandas"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit-learn": version("scikit-learn"),
    }
    report_payloads: dict[Path, bytes] = {
        artifacts.splits: manifest_bytes,
        artifacts.split_summary: split_summary_bytes,
        artifacts.validation_predictions: predictions_bytes,
        artifacts.metrics: metrics_bytes,
        artifacts.config: config_bytes,
    }
    artifact_paths = (*report_payloads, artifacts.provenance)
    if len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError("Day03 output paths must be unique")
    if set(artifact_paths).intersection(input_bytes):
        raise ValueError("Day03 output paths must not overwrite inputs")
    provenance_payload = {
        "source": {key: source[key] for key in ("dataset", "revision", "subset") if key in source},
        "preprocessing_policy": day02_provenance.get("preprocessing_policy", PREPROCESSING_POLICY),
        "day02_provenance_sha256": _sha256(input_bytes[project_root / config.provenance_path]),
        "input_checksums": {
            str(path.relative_to(project_root)): _sha256(content)
            for path, content in input_bytes.items()
        },
        "checksums": {
            "config": _sha256(input_bytes[resolved_config_path]),
            "effective_config": _sha256(config_bytes),
            "processed_csv": _sha256(input_bytes[project_root / config.input_path]),
            "manifest": _sha256(manifest_bytes),
            "split_summary": _sha256(split_summary_bytes),
            "predictions": _sha256(predictions_bytes),
            "metrics": _sha256(metrics_bytes),
        },
        "random_state": config.random_state,
        "split_algorithm": splits.summary.algorithm,
        "vocabulary_size": fit.feature_count,
        "classes": {
            "class_order": list(CLASS_ORDER),
            "label_mapping": LABEL_MAPPING,
            "logreg_classes": fit.pipeline.named_steps["clf"].classes_.tolist(),
            "dummy_classes": [int(value) for value in cast(np.ndarray, fit.dummy.classes_)],
        },
        "versions": versions,
    }
    provenance_bytes = _json_bytes(provenance_payload)
    if json.loads(provenance_bytes) != provenance_payload:
        raise ValueError("Generated provenance failed its JSON round-trip check")

    changed_inputs = [path for path, content in input_bytes.items() if path.read_bytes() != content]
    if changed_inputs:
        raise ValueError(f"Input bytes changed during Day03 run; outputs not written: {changed_inputs}")
    for path, content in report_payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    artifacts.provenance.parent.mkdir(parents=True, exist_ok=True)
    artifacts.provenance.write_bytes(provenance_bytes)
    return BaselineRun(config=config, fit=fit, evaluation=evaluation, artifacts=artifacts)
