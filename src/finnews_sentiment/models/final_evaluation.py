"""Explicit frozen test evaluation; completed results are verified and reused."""

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from finnews_sentiment.data.baseline_config import load_baseline_config
from finnews_sentiment.data.experiment_splits import Day04Inputs, load_day04_inputs
from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.features.labels_duplicates import LABEL_MAPPING
from finnews_sentiment.models.confusion import (
    ValidationConfusion, build_validation_confusion, plot_validation_confusion,
)
from finnews_sentiment.models.error_analysis import ValidationAnalysis, load_validation_analysis
from finnews_sentiment.models.evaluate import _label_ids
from finnews_sentiment.models.run_experiments_day04 import _candidate_pipeline, _classification_payload

FINAL_NAMES = (
    "frozen_protocol.json", "test_predictions.csv", "test_metrics.json",
    "test_confusion_counts.csv", "test_confusion_counts.png",
    "test_confusion_normalized.csv", "test_confusion_normalized.png",
    "final_evaluation.json",
)
PREPARE_COMMAND = "PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.final_evaluation"


@dataclass(frozen=True)
class FinalEvaluation:
    """Typed notebook/report return contract, independent of kernel narrowing."""

    cache_status: Literal["created", "reused"]
    candidate_id: str
    protocol: dict[str, object]
    predictions: pd.DataFrame
    confusion: ValidationConfusion
    n_train: int
    n_features: int
    directory: Path


@dataclass(frozen=True)
class _Reference:
    root: Path
    analysis: ValidationAnalysis
    inputs: Day04Inputs
    protocol: dict[str, object]
    hashes: dict[str, str]


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_bytes())
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def runtime_versions() -> dict[str, str]:
    """Actual versions used by fitting, metrics, rendering and numerical runtime."""
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in (
            "pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl",
        )},
    }


def _prepare(root: Path) -> _Reference:
    analysis = load_validation_analysis(root)
    inputs = load_day04_inputs(root)
    config = load_baseline_config(root / "configs/train_baseline.json")
    provenance = _read_json(root / "reports/day04/provenance.json")
    versions = runtime_versions()
    for path in (root / "reports/day04/provenance.json", root / config.report_output_dir / "provenance.json"):
        saved = _read_json(path).get("versions")
        expected = {name: versions[name] for name in ("python", "pandas", "numpy", "scipy", "scikit-learn")}
        if saved != expected:
            raise ValueError(f"Runtime version mismatch: {path}; expected {expected}, recorded {saved}")
    recorded_inputs = provenance.get("input_checksums")
    recorded_outputs = provenance.get("output_checksums")
    if not isinstance(recorded_inputs, dict) or not isinstance(recorded_outputs, dict):
        raise ValueError("Invalid Day04 checksum maps")
    paths = [root / name for name in recorded_inputs]
    paths.extend(root / "reports/day04" / name for name in recorded_outputs)
    paths.append(root / "reports/day04/provenance.json")
    hashes = {str(path.relative_to(root)): _sha(path.read_bytes()) for path in paths}
    day02 = _read_json(root / config.provenance_path)
    train_ids = inputs.splits["train"]["source_row_id"].tolist()
    pipeline, _ = _candidate_pipeline(analysis.candidate_id, config)
    tfidf_parameters = pipeline.named_steps["tfidf"].get_params(deep=False)
    # NumPy dtype classes are serialized by their canonical dtype name.
    tfidf_parameters["dtype"] = np.dtype(tfidf_parameters["dtype"]).name
    protocol: dict[str, object] = {
        "policy": "day05-frozen-protocol-v1",
        "candidate_id": analysis.candidate_id, "parameters": analysis.parameters,
        "pipeline_parameters": {"tfidf": tfidf_parameters,
                                "classifier": pipeline.named_steps["clf"].get_params(deep=False)},
        "preprocessing_policy": day02.get("preprocessing_policy"),
        "preprocessing_parameters": day02.get("parameters"),
        "fit_policy": "day03_train_only", "text_col": config.text_col,
        "target_col": config.target_col, "random_state": config.random_state,
        "class_order": list(CLASS_ORDER), "label_mapping": LABEL_MAPPING,
        "metric_policy": {"labels": [0, 1, 2], "average": "macro", "zero_division": 0},
        "train_source_row_ids": train_ids,
        "validation_source_row_ids": inputs.splits["validation"]["source_row_id"].tolist(),
        "test_source_row_ids": inputs.manifest.loc[inputs.manifest["split"].eq("test"), "source_row_id"].tolist(),
        "input_checksums": hashes, "runtime_versions": versions,
        "validation_replay_policy": "exact IDs, truth labels, order and selected predictions before test predict",
    }
    return _Reference(root, analysis, inputs, protocol, hashes)


def _unchanged(reference: _Reference) -> None:
    for name, expected in reference.hashes.items():
        if _sha((reference.root / name).read_bytes()) != expected:
            raise ValueError(f"Input changed during final evaluation: {name}")


def _test_metrics(reference: _Reference, summary: ValidationConfusion) -> dict[str, object]:
    return {
        "evaluation_split": "test", "n_test": summary.metrics.n_samples,
        "candidate_id": reference.analysis.candidate_id, "fit_policy": "day03_train_only",
        "class_order": list(CLASS_ORDER), "label_mapping": LABEL_MAPPING,
        **_classification_payload(summary.metrics),
    }


def _read_predictions(content: bytes, reference: _Reference) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(content), dtype="str", keep_default_na=False)
    columns = ("source_row_id", "true_label_id", "pred_label_id")
    if tuple(frame.columns) != columns:
        raise ValueError(f"Test prediction schema must be {columns}")
    for name in columns:
        if not frame[name].str.fullmatch(r"[0-9]+", na=False).all():
            raise ValueError(f"Invalid integer literals in test predictions: {name}")
    frame = frame.astype("int64")
    test = reference.inputs.splits["test"]
    if frame["source_row_id"].duplicated().any() or frame["source_row_id"].tolist() != test["source_row_id"].tolist():
        raise ValueError("Test prediction IDs/order differ from frozen manifest")
    if frame["true_label_id"].tolist() != test["label_id"].tolist():
        raise ValueError("Test true labels differ from processed reference")
    _label_ids(frame["pred_label_id"].to_numpy(), "cached test predictions")
    return frame


def _validate_payloads(payloads: dict[str, bytes], reference: _Reference) -> tuple[pd.DataFrame, ValidationConfusion]:
    predictions = _read_predictions(payloads["test_predictions.csv"], reference)
    summary = build_validation_confusion(predictions)
    if json.loads(payloads["test_metrics.json"]) != _test_metrics(reference, summary):
        raise ValueError("Cached test metrics do not reproduce predictions")
    for kind, expected in (("counts", summary.counts), ("normalized", summary.normalized)):
        frame = pd.read_csv(BytesIO(payloads[f"test_confusion_{kind}.csv"]), index_col="true_label", float_precision="round_trip")
        frame.columns.name = "pred_label"
        try:
            pd.testing.assert_frame_equal(frame, expected, check_exact=True)
        except AssertionError as error:
            raise ValueError(f"Cached test {kind} matrix differs from predictions") from error
        if not payloads[f"test_confusion_{kind}.png"].startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Invalid cached test {kind} PNG")
    return predictions, summary


def _reuse(reference: _Reference) -> FinalEvaluation:
    directory = reference.root / "reports/day05"
    missing = [name for name in FINAL_NAMES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete final cache: missing {missing}")
    protocol_bytes = (directory / "frozen_protocol.json").read_bytes()
    if protocol_bytes != _json_bytes(reference.protocol):
        raise ValueError("Frozen protocol mismatch (inputs/config/selection/runtime); no re-evaluation")
    marker = _read_json(directory / "final_evaluation.json")
    if marker.get("policy") != "day05-final-cache-v1" or marker.get("protocol_sha256") != _sha(protocol_bytes):
        raise ValueError("Final cache marker policy/protocol checksum mismatch")
    hashes = marker.get("output_checksums")
    output_names = set(FINAL_NAMES) - {"frozen_protocol.json", "final_evaluation.json"}
    if not isinstance(hashes, dict) or set(hashes) != output_names:
        raise ValueError("Final cache output checksum schema mismatch")
    payloads: dict[str, bytes] = {}
    for name in sorted(output_names):
        content = (directory / name).read_bytes()
        if _sha(content) != hashes[name]:
            raise ValueError(f"Final cache checksum mismatch: {directory / name}")
        payloads[name] = content
    if marker.get("runtime_versions") != reference.protocol["runtime_versions"]:
        raise ValueError("Final cache runtime mismatch")
    replay = marker.get("validation_replay")
    if not isinstance(replay, dict):
        raise ValueError("Final cache must record validation replay")
    n_features = replay.get("n_features")
    if (replay.get("status") != "exact_match"
            or replay.get("n_validation") != len(reference.analysis.table)
            or replay.get("n_train") != len(reference.inputs.splits["train"])
            or not isinstance(n_features, int) or isinstance(n_features, bool) or n_features <= 0):
        raise ValueError("Final cache replay metadata mismatch")
    predictions, summary = _validate_payloads(payloads, reference)
    _unchanged(reference)
    return FinalEvaluation("reused", reference.analysis.candidate_id, reference.protocol,
                           predictions, summary, len(reference.inputs.splits["train"]), n_features, directory)


def _cache_error(root: Path, error: Exception) -> ValueError:
    return ValueError(
        f"Final evaluation stopped under {root / 'reports/day05'}: {error}. "
        "Inspect frozen_protocol.json, final_evaluation.json and input/output hashes. "
        "No automatic overwrite, deletion or new test evaluation is permitted. "
        "Partial artifacts require explicit diagnosis before any recovery."
    )


def load_final_evaluation(root: Path) -> FinalEvaluation:
    """Read and recheck a completed final cache; never fit, predict or write."""
    root = root.resolve()
    if not (root / "reports/day05/final_evaluation.json").exists():
        raise ValueError(f"Missing completed final cache under {root / 'reports/day05'}. Run explicitly: {PREPARE_COMMAND}; inspect any partial artifacts first.")
    try:
        return _reuse(_prepare(root))
    except (OSError, ValueError) as error:
        raise _cache_error(root, error) from error


def run_final_evaluation(root: Path) -> FinalEvaluation:
    """Freeze, fit train, replay validation, predict test once; else reuse cache.

    Exclusive protocol creation reserves the first run. Completion marker is
    written last. A failure leaves partial artifacts and blocks automatic retry;
    exactly-once prediction across a process crash cannot be guaranteed.
    """
    root = root.resolve()
    directory = root / "reports/day05"
    try:
        reference = _prepare(root)
        if any((directory / name).exists() for name in FINAL_NAMES):
            return _reuse(reference)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "frozen_protocol.json").open("xb") as stream:
            stream.write(_json_bytes(reference.protocol))
        config = load_baseline_config(root / "configs/train_baseline.json")
        pipeline, selected_config = _candidate_pipeline(reference.analysis.candidate_id, config)
        train = reference.inputs.splits["train"]
        validation = reference.inputs.splits["validation"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(train[selected_config.text_col].tolist(), train[selected_config.target_col].to_numpy())
            replay = _label_ids(np.asarray(pipeline.predict(validation[selected_config.text_col].tolist())), "validation replay")
            # IDs and truth are verified by D5-01, not inferred from array length.
            if (validation["source_row_id"].tolist() != reference.analysis.table["source_row_id"].tolist()
                    or validation["label_id"].tolist() != reference.analysis.table["true_label_id"].tolist()
                    or not np.array_equal(replay, reference.analysis.table["pred_label_id"].to_numpy())):
                raise ValueError("Validation replay mismatch; test predict was not called")
            _unchanged(reference)
            test = reference.inputs.splits["test"]
            predicted = _label_ids(np.asarray(pipeline.predict(test[selected_config.text_col].tolist())), "test predictions")
        if len(predicted) != len(test):
            raise ValueError("Test prediction length mismatch")
        predictions = pd.DataFrame({
            "source_row_id": test["source_row_id"].to_numpy(),
            "true_label_id": test["label_id"].to_numpy(), "pred_label_id": predicted,
        })
        n_features = len(pipeline.named_steps["tfidf"].get_feature_names_out())
        summary = build_validation_confusion(predictions)
        payloads = {
            "test_predictions.csv": predictions.to_csv(index=False, lineterminator="\n").encode("utf-8"),
            "test_metrics.json": _json_bytes(_test_metrics(reference, summary)),
        }
        for kind, frame in (("counts", summary.counts), ("normalized", summary.normalized)):
            payloads[f"test_confusion_{kind}.csv"] = frame.to_csv(lineterminator="\n").encode("utf-8")
            figure = plot_validation_confusion(summary, normalized=kind == "normalized", evaluation_split="test")
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=140)
            figure.clear()
            payloads[f"test_confusion_{kind}.png"] = buffer.getvalue()
        _validate_payloads(payloads, reference)
        _unchanged(reference)
        for name, content in payloads.items():
            with (directory / name).open("xb") as stream:
                stream.write(content)
        marker = {
            "policy": "day05-final-cache-v1", "protocol_sha256": _sha(_json_bytes(reference.protocol)),
            "runtime_versions": reference.protocol["runtime_versions"],
            "validation_replay": {"status": "exact_match", "n_train": len(train),
                                  "n_validation": len(validation), "n_features": n_features},
            "output_checksums": {name: _sha(content) for name, content in payloads.items()},
        }
        # All final payloads must have reached disk intact before completion.
        for name, content in payloads.items():
            if (directory / name).read_bytes() != content:
                raise ValueError(f"Final output read-back mismatch: {name}")
        _unchanged(reference)
        with (directory / "final_evaluation.json").open("xb") as stream:
            stream.write(_json_bytes(marker))
        return FinalEvaluation("created", reference.analysis.candidate_id, reference.protocol,
                               predictions, summary, len(train), n_features, directory)
    except (OSError, ValueError, ConvergenceWarning) as error:
        raise _cache_error(root, error) from error


def main() -> None:
    """Explicit final-evaluation command, distinct from future report-only CLI."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    result = run_final_evaluation(args.root)
    print(f"Cache: {result.cache_status}; candidate: {result.candidate_id}; validation replay: exact_match")
    print(f"TF-IDF train shape: ({result.n_train}, {result.n_features})")
    print(f"Test N={len(result.predictions)}; accuracy={result.confusion.metrics.accuracy:.6f}; macro F1={result.confusion.metrics.macro_f1:.6f}")
    print(result.confusion.per_class.to_string(index=False))
    print(f"Final cache: {result.directory / 'final_evaluation.json'}")


if __name__ == "__main__":
    main()
