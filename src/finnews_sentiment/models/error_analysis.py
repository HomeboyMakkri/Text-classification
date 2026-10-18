"""Read verified validation references without fitting or predicting."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd

from finnews_sentiment.data.experiment_splits import load_day04_inputs, _object, _read_object
from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.models.run_experiments_day04 import (
    _candidate_parameters, _candidate_pipeline,
    _expected_experiment_config, _metrics_payload, select_cv_winner, TIE_ORDER,
)
from finnews_sentiment.data.baseline_config import load_baseline_config
from finnews_sentiment.models.evaluate import evaluate_predictions

PREDICTION_COLUMNS = (
    "source_row_id", "true_label_id", "selected_pred_label_id", "day03_baseline_pred_label_id",
)


@dataclass(frozen=True)
class ValidationAnalysis:
    """All validation rows and their ordered error subset, plus selection source."""

    table: pd.DataFrame
    errors: pd.DataFrame
    candidate_id: str
    parameters: dict[str, object]
    selection_path: Path


def build_validation_table(
    validation: pd.DataFrame, manifest: pd.DataFrame, predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Join one-to-one by source ID; reject reordered or corrupted references."""
    required = ("source_row_id", "text", "text_clean", "label_id")
    if any(column not in validation for column in required):
        raise ValueError(f"Validation must contain {required}")
    if tuple(manifest.columns) != ("source_row_id", "group_id", "split"):
        raise ValueError("Manifest schema must be source_row_id, group_id, split")
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"Day04 predictions columns must be {PREDICTION_COLUMNS}")
    if validation.empty or validation.isna().any().any() or manifest.isna().any().any():
        raise ValueError("Validation/manifest must be nonempty and contain no missing values")
    for column in ("text", "text_clean"):
        if not validation[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Validation {column} must contain nonempty strings")
    for frame in (validation, manifest, predictions):
        if frame["source_row_id"].duplicated().any():
            raise ValueError("source_row_id must be unique")
    for column in PREDICTION_COLUMNS:
        values = predictions[column].astype("string")
        if not values.str.fullmatch(r"[0-9]+", na=False).all():
            raise ValueError(f"Prediction {column} must contain integer literals without missing values")
    predictions = predictions.astype("int64")
    for column in PREDICTION_COLUMNS[1:]:
        if not predictions[column].isin((0, 1, 2)).all():
            raise ValueError(f"Prediction {column} contains unknown labels")
    valid_manifest = manifest.loc[manifest["split"].eq("validation")]
    ids = valid_manifest["source_row_id"].tolist()
    if validation["source_row_id"].tolist() != ids or predictions["source_row_id"].tolist() != ids:
        raise ValueError("Validation source IDs must have exact coverage and manifest order")
    other = manifest.loc[~manifest["split"].eq("validation")]
    if not set(valid_manifest["group_id"]).isdisjoint(other["group_id"]):
        raise ValueError("Validation groups overlap train/test")
    if not validation["label_id"].isin((0, 1, 2)).all():
        raise ValueError("Validation contains unknown labels")
    joined = valid_manifest[["source_row_id", "group_id"]].merge(
        validation[list(required)], on="source_row_id", how="left", validate="one_to_one", sort=False,
    ).merge(predictions, on="source_row_id", how="left", validate="one_to_one", sort=False)
    if not joined["label_id"].eq(joined["true_label_id"]).all():
        raise ValueError("Prediction true labels differ from processed validation labels")
    joined = joined.rename(columns={"selected_pred_label_id": "pred_label_id"})
    mapping = dict(enumerate(CLASS_ORDER))
    joined["true_label"] = joined["true_label_id"].map(mapping)
    joined["pred_label"] = joined["pred_label_id"].map(mapping)
    joined["is_error"] = joined["true_label_id"].ne(joined["pred_label_id"])
    return joined[[
        "source_row_id", "group_id", "text", "text_clean", "true_label_id",
        "pred_label_id", "true_label", "pred_label", "is_error",
    ]].copy()


def load_validation_analysis(root: Path) -> ValidationAnalysis:
    """Verify existing Day02–Day04 artifacts and return only validation analysis.

    No files are written. Missing references require explicit preparation;
    the loader never starts the Day04 training command itself.
    """
    root = root.resolve()
    directory = root / "reports/day04"
    command = "PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.run_experiments_day04"
    try:
        provenance = _read_object(directory / "provenance.json", "Day04 provenance")
        if provenance.get("policy") != "day04-provenance-v1" or provenance.get("class_order") != list(CLASS_ORDER):
            raise ValueError("Day04 provenance policy/class order mismatch")
        input_hashes = _object(provenance.get("input_checksums"), "Day04 input_checksums")
        output_hashes = _object(provenance.get("output_checksums"), "Day04 output_checksums")
        config = load_baseline_config(root / "configs/train_baseline.json")
        required_inputs = {
            "configs/experiments_day04.json", "configs/train_baseline.json", "configs/dataset.json",
            str(config.input_path), str(config.provenance_path),
            str(config.split_output_dir / "splits.csv"),
            *(str(config.report_output_dir / name) for name in (
                "split_summary.json", "validation_predictions.csv", "metrics.json", "config.json", "provenance.json",
            )),
        }
        required_outputs = {
            "cv_folds.csv", "cv_fold_summary.csv", "cv_results.csv", "cv_summary.csv",
            "effective_config.json", "selection.json", "validation_predictions.csv", "metrics.json",
        }
        if set(input_hashes) != required_inputs or set(output_hashes) != required_outputs:
            raise ValueError("Day04 checksum schema does not cover required inputs/outputs")
        contents: dict[str, bytes] = {}
        for hashes, base in ((input_hashes, root), (output_hashes, directory)):
            for name, expected in hashes.items():
                path = base / name
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"Day04 checksum mismatch: {path}")
                if base == directory:
                    contents[name] = content
        source = _object(provenance.get("source"), "Day04 source")
        if source.get("day02_provenance_sha256") != input_hashes[str(config.provenance_path)]:
            raise ValueError("Day04 Day02 provenance mismatch")
        if provenance.get("day03_provenance_sha256") != input_hashes[str(config.report_output_dir / "provenance.json")]:
            raise ValueError("Day04 Day03 provenance mismatch")
        if _read_object(root / "configs/experiments_day04.json", "Day04 experiment config") != _expected_experiment_config():
            raise ValueError("Day04 experiment config mismatch")
        inputs = load_day04_inputs(root)
        selection = _read_object(directory / "selection.json", "Day04 selection")
        effective = _read_object(directory / "effective_config.json", "Day04 effective config")
        metrics = _read_object(directory / "metrics.json", "Day04 metrics")
        candidate = selection.get("candidate_id")
        if not isinstance(candidate, str):
            raise ValueError("Day04 selection candidate_id must be a string")
        pipeline, candidate_config = _candidate_pipeline(candidate, config)
        parameters = _object(selection.get("parameters"), "Day04 selected parameters")
        if parameters != _candidate_parameters(pipeline, candidate_config):
            raise ValueError("Day04 selection parameters differ from saved candidate config")
        expected_effective = {
            "policy": "day04-effective-config-v1", "experiment_config": _expected_experiment_config(),
            "baseline_config": {"policy": config.policy, "tfidf": {**config.tfidf, "ngram_range": list(config.tfidf["ngram_range"])}, "logreg": dict(config.logreg)},
            "folds": {"n_splits": 3, "random_state": 42}, "selected_candidate_id": candidate,
        }
        if provenance.get("cv") != expected_effective["folds"]:
            raise ValueError("Day04 provenance CV configuration mismatch")
        if effective != expected_effective:
            raise ValueError("Day04 effective config mismatch")
        summary = pd.read_csv(BytesIO(contents["cv_summary.csv"]), float_precision="round_trip")
        winner, tied = select_cv_winner(summary)
        selected_mean = float(summary.loc[summary["candidate_id"].eq(candidate), "mean_macro_f1"].iloc[0])
        if (selection.get("selection_metric") != "mean_macro_f1"
                or selection.get("tie_order") != list(TIE_ORDER)
                or selection.get("selected_mean_cv_macro_f1") != selected_mean):
            raise ValueError("Day04 selection metric/score/tie order mismatch")
        if winner != candidate or selection.get("tie_candidates_at_max") != list(tied):
            raise ValueError("Day04 selection differs from saved CV rule")
        if selection.get("held_out_test_used") is not False or selection.get("outer_validation_used_for_selection") is not False:
            raise ValueError("Day04 selection violates validation/test boundary")
        if _object(provenance.get("selection"), "Day04 provenance selection") != {"candidate_id": candidate, "metric": "mean_macro_f1"}:
            raise ValueError("Day04 provenance selection mismatch")
        predictions = pd.read_csv(BytesIO(contents["validation_predictions.csv"]), dtype="str", keep_default_na=False)
        table = build_validation_table(inputs.splits["validation"], inputs.manifest, predictions)
        if predictions["day03_baseline_pred_label_id"].astype("int64").tolist() != inputs.validation_predictions["baseline_pred_label_id"].tolist():
            raise ValueError("Day04 baseline predictions differ from Day03 reference")
        selected_metrics = evaluate_predictions(table["true_label_id"].to_numpy(), table["pred_label_id"].to_numpy())
        baseline_metrics = evaluate_predictions(table["true_label_id"].to_numpy(), predictions["day03_baseline_pred_label_id"].astype("int64").to_numpy())
        expected_metrics = _metrics_payload(candidate, selected_metrics, baseline_metrics)
        confirmation_warnings = metrics.get("confirmation_warnings")
        if not isinstance(confirmation_warnings, list) or any(not isinstance(item, str) for item in confirmation_warnings):
            raise ValueError("Day04 confirmation_warnings must be a list of strings")
        expected_metrics["confirmation_warnings"] = confirmation_warnings
        if metrics != expected_metrics:
            raise ValueError("Day04 metrics do not reproduce saved validation predictions")
        return ValidationAnalysis(table, table.loc[table["is_error"]].copy(), candidate, parameters, directory / "selection.json")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing reference: {error.filename or error}. Prepare Day04 explicitly: {command}") from error
    except ValueError as error:
        raise ValueError(f"Invalid validation reference under {directory}: {error}") from error
