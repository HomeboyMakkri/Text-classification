"""Run, select and validate the fixed Day04 candidate set offline."""

from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from finnews_sentiment.data.baseline_config import BaselineConfig, load_baseline_config
from finnews_sentiment.data.experiment_splits import CVFoldSet, load_day04_inputs, build_train_cv_folds
from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.models.cv_runner import CVRunResult, run_cv_candidate
from finnews_sentiment.models.cv_search import CSearchResult, search_logreg_c
from finnews_sentiment.models.evaluate import ClassificationMetrics, evaluate_predictions
from finnews_sentiment.models.experiment_models import build_linear_svc_pipeline
from finnews_sentiment.models.train_model import build_baseline_pipeline


EXPERIMENT_CONFIG = Path("configs/experiments_day04.json")
TIE_ORDER = (
    "baseline", "word_unigram", "linear_svc", "logreg_c_0_1", "logreg_c_1_0", "logreg_c_10_0",
)
LOGREG_C_CANDIDATES = {
    0.1: "logreg_c_0_1",
    1.0: "logreg_c_1_0",
    10.0: "logreg_c_10_0",
}


@dataclass(frozen=True)
class Day04ArtifactPaths:
    """Paths written by the final Day04 comparison run."""

    fold_manifest: Path
    fold_summary: Path
    cv_results: Path
    cv_summary: Path
    effective_config: Path
    selection: Path
    validation_predictions: Path
    metrics: Path
    provenance: Path


@dataclass(frozen=True)
class Day04Run:
    """Comparison, chosen candidate and its one-time validation confirmation."""

    fold_set: CVFoldSet
    cv_results: pd.DataFrame
    cv_summary: pd.DataFrame
    selection: dict[str, object]
    validation_predictions: pd.DataFrame
    metrics: dict[str, object]
    artifacts: Day04ArtifactPaths


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _expected_experiment_config() -> dict[str, object]:
    return {
        "policy": "day04-fixed-experiments-v1",
        "baseline_config": "configs/train_baseline.json",
        "cv": {
            "splitter": "StratifiedGroupKFold", "n_splits": 3, "shuffle": True,
            "random_state": 42, "scope": "day03_train_only",
        },
        "selection": {
            "metric": "mean_macro_f1", "labels": [0, 1, 2], "zero_division": 0,
            "tie_order": list(TIE_ORDER),
        },
        "candidates": {
            "baseline": {},
            "word_unigram": {"tfidf_overrides": {"ngram_range": [1, 1]}},
            "linear_svc": {"classifier": {
                "penalty": "l2", "loss": "squared_hinge", "dual": "auto", "tol": 1e-4,
                "C": 1.0, "multi_class": "ovr", "fit_intercept": True,
                "intercept_scaling": 1, "class_weight": None, "verbose": 0,
                "random_state": 42, "max_iter": 5000,
            }},
            "logreg_C": [0.1, 1.0, 10.0],
        },
    }


def _load_experiment_config(path: Path) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load Day04 experiment config {path}: {error}") from error
    if payload != _expected_experiment_config():
        raise ValueError("Day04 experiment config must match the frozen SPEC candidate grid and selection rule")
    if not isinstance(payload, dict):
        raise ValueError("Day04 experiment config must be a JSON object")
    return payload


def _candidate_pipeline(candidate_id: str, config: BaselineConfig) -> tuple[Pipeline, BaselineConfig]:
    if candidate_id == "baseline":
        return build_baseline_pipeline(config), config
    if candidate_id == "word_unigram":
        unigram_config = replace(config, tfidf={**config.tfidf, "ngram_range": (1, 1)})
        return build_baseline_pipeline(unigram_config), unigram_config
    if candidate_id == "linear_svc":
        return build_linear_svc_pipeline(config), config
    candidate_c = next((value for value, name in LOGREG_C_CANDIDATES.items() if name == candidate_id), None)
    if candidate_c is None:
        raise ValueError(f"Unknown Day04 candidate_id {candidate_id!r}")
    candidate_config = replace(config, logreg={**config.logreg, "C": candidate_c})
    return build_baseline_pipeline(candidate_config), candidate_config


def _candidate_parameters(pipeline: Pipeline, config: BaselineConfig) -> dict[str, object]:
    vectorizer = pipeline.named_steps.get("tfidf")
    classifier = pipeline.named_steps.get("clf")
    if vectorizer is None or classifier is None:
        raise ValueError("Candidate Pipeline must have 'tfidf' and 'clf' steps")
    tfidf = {**config.tfidf, "ngram_range": list(config.tfidf["ngram_range"])}
    classifier_parameters = classifier.get_params(deep=False)
    return {"tfidf": tfidf, "classifier": classifier_parameters}


def _fold_rows(result: CVRunResult, fold_set: CVFoldSet) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold_id in range(result.n_splits):
        score = result.fold_scores.iloc[fold_id]
        summary = fold_set.summary.iloc[fold_id]
        rows.append({
            "candidate_id": result.candidate_id,
            "fold_id": fold_id,
            "train_rows": int(score["train_rows"]),
            "train_groups": int(score["train_groups"]),
            "validation_rows": int(score["validation_rows"]),
            "validation_groups": int(score["validation_groups"]),
            "n_features": int(score["n_features"]),
            "macro_f1": float(score["macro_f1"]),
            "accuracy": float(score["accuracy"]),
            "convergence_status": str(score["convergence_status"]),
            "warnings": " | ".join(score["warnings"]),
            "fold_train_rows_expected": int(summary["train_rows"]),
            "fold_validation_rows_expected": int(summary["validation_rows"]),
        })
    return rows


def _summary_from_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_scores = results.loc[results["candidate_id"].eq("baseline")]
    baseline_macro_f1 = float(baseline_scores["macro_f1"].mean())
    baseline_accuracy = float(baseline_scores["accuracy"].mean())
    for candidate_id in TIE_ORDER:
        candidate = results.loc[results["candidate_id"].eq(candidate_id)]
        if candidate.empty:
            raise ValueError(f"CV results are missing candidate {candidate_id!r}")
        rows.append({
            "candidate_id": candidate_id,
            "mean_macro_f1": float(candidate["macro_f1"].mean()),
            "std_macro_f1": float(candidate["macro_f1"].std(ddof=0)),
            "mean_accuracy": float(candidate["accuracy"].mean()),
            "std_accuracy": float(candidate["accuracy"].std(ddof=0)),
            "delta_macro_f1": float(candidate["macro_f1"].mean()) - baseline_macro_f1,
            "delta_accuracy": float(candidate["accuracy"].mean()) - baseline_accuracy,
            "n_features_min": int(candidate["n_features"].min()),
            "n_features_max": int(candidate["n_features"].max()),
        })
    return pd.DataFrame(rows)


def select_cv_winner(cv_summary: pd.DataFrame) -> tuple[str, tuple[str, ...]]:
    """Select max mean CV macro F1 with the frozen Day04 exact-tie order."""
    required_columns = ("candidate_id", "mean_macro_f1")
    if tuple(column for column in required_columns if column in cv_summary.columns) != required_columns:
        raise ValueError(f"CV summary must contain {required_columns}")
    if cv_summary["candidate_id"].duplicated().any():
        raise ValueError("CV summary candidate IDs must be unique")
    scores = dict(zip(cv_summary["candidate_id"], cv_summary["mean_macro_f1"], strict=True))
    if set(scores) != set(TIE_ORDER) or not np.isfinite(list(scores.values())).all():
        raise ValueError("CV summary must contain finite mean macro F1 for every fixed candidate")
    best_score = max(float(value) for value in scores.values())
    tied = tuple(candidate_id for candidate_id in TIE_ORDER if float(scores[candidate_id]) == best_score)
    return tied[0], tied


def _classification_payload(metrics: ClassificationMetrics) -> dict[str, object]:
    payload = asdict(metrics)
    payload["per_class"] = [asdict(item) for item in metrics.per_class]
    return payload


def _metrics_payload(
    selected_candidate_id: str,
    selected: ClassificationMetrics,
    baseline: ClassificationMetrics,
) -> dict[str, object]:
    delta_macro_f1 = selected.macro_f1 - baseline.macro_f1
    relative_delta = delta_macro_f1 / baseline.macro_f1 * 100 if baseline.macro_f1 > 0 else None
    return {
        "evaluation_split": "validation",
        "selected_candidate_id": selected_candidate_id,
        "class_order": list(CLASS_ORDER),
        "selected": _classification_payload(selected),
        "day03_baseline": _classification_payload(baseline),
        "delta": {
            "macro_f1_absolute": delta_macro_f1,
            "macro_f1_relative_percent": relative_delta,
            "accuracy_absolute": selected.accuracy - baseline.accuracy,
        },
    }


def _roundtrip_csv(content: bytes, *, int_columns: tuple[str, ...], columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            BytesIO(content), encoding="utf-8", keep_default_na=False,
            dtype={column: "int64" for column in int_columns},
        )
    except (ValueError, pd.errors.ParserError) as error:
        raise ValueError("Generated Day04 CSV failed its read-back check") from error
    if tuple(frame.columns) != columns:
        raise ValueError(f"Generated Day04 CSV columns differ from {columns}")
    return frame


def _artifact_paths(root: Path) -> Day04ArtifactPaths:
    directory = root / "reports/day04"
    return Day04ArtifactPaths(
        fold_manifest=directory / "cv_folds.csv",
        fold_summary=directory / "cv_fold_summary.csv",
        cv_results=directory / "cv_results.csv",
        cv_summary=directory / "cv_summary.csv",
        effective_config=directory / "effective_config.json",
        selection=directory / "selection.json",
        validation_predictions=directory / "validation_predictions.csv",
        metrics=directory / "metrics.json",
        provenance=directory / "provenance.json",
    )


def run_experiments_day04(
    experiment_config_path: Path = EXPERIMENT_CONFIG,
    root: Path | None = None,
) -> Day04Run:
    """Run all fixed CV candidates, select by CV, then confirm once on validation."""
    project_root = (root or Path(__file__).resolve().parents[3]).resolve()
    resolved_experiment_path = experiment_config_path
    if not resolved_experiment_path.is_absolute():
        resolved_experiment_path = project_root / resolved_experiment_path
    resolved_experiment_path = resolved_experiment_path.resolve()
    experiment_config = _load_experiment_config(resolved_experiment_path)
    baseline_config_path = project_root / "configs/train_baseline.json"
    baseline_config = load_baseline_config(baseline_config_path)
    config_source_path = project_root / baseline_config.provenance_path
    processed_path = project_root / baseline_config.input_path
    day03_report = project_root / baseline_config.report_output_dir
    input_paths = (
        resolved_experiment_path,
        baseline_config_path,
        project_root / "configs/dataset.json",
        processed_path,
        config_source_path,
        project_root / baseline_config.split_output_dir / "splits.csv",
        day03_report / "split_summary.json",
        day03_report / "validation_predictions.csv",
        day03_report / "metrics.json",
        day03_report / "config.json",
        day03_report / "provenance.json",
    )
    input_bytes = {path: path.read_bytes() for path in input_paths}
    day04_inputs = load_day04_inputs(project_root)
    train = day04_inputs.splits["train"]
    validation = day04_inputs.splits["validation"]
    fold_set = build_train_cv_folds(train, day04_inputs.train_manifest)

    candidate_results: list[CVRunResult] = []
    candidate_parameter_sets: dict[str, dict[str, object]] = {}
    for candidate_id in TIE_ORDER[:3]:
        pipeline, candidate_config = _candidate_pipeline(candidate_id, baseline_config)
        candidate_results.append(run_cv_candidate(candidate_id, pipeline, train, fold_set))
        candidate_parameter_sets[candidate_id] = _candidate_parameters(pipeline, candidate_config)

    c_search: CSearchResult = search_logreg_c(build_baseline_pipeline(baseline_config), train, fold_set)
    baseline_c_scores = next(result for result in candidate_results if result.candidate_id == "baseline")
    c_one_scores = c_search.scores.loc[c_search.scores["C"].eq(1.0)].iloc[0]
    for fold_id in range(fold_set.n_splits):
        baseline_fold = baseline_c_scores.fold_scores.iloc[fold_id]
        if not np.isclose(float(c_one_scores[f"fold_{fold_id}_macro_f1"]), float(baseline_fold["macro_f1"])):
            raise RuntimeError("GridSearchCV C=1 macro F1 differs from the D4-03 baseline control")
        if not np.isclose(float(c_one_scores[f"fold_{fold_id}_accuracy"]), float(baseline_fold["accuracy"])):
            raise RuntimeError("GridSearchCV C=1 accuracy differs from the D4-03 baseline control")

    baseline_features = baseline_c_scores.fold_scores["n_features"].to_numpy(dtype=np.int64)
    c_rows: list[dict[str, object]] = []
    for _, c_score in c_search.scores.iterrows():
        value = float(c_score["C"])
        candidate_id = LOGREG_C_CANDIDATES[value]
        candidate_config = replace(baseline_config, logreg={**baseline_config.logreg, "C": value})
        c_pipeline = build_baseline_pipeline(candidate_config)
        candidate_parameter_sets[candidate_id] = _candidate_parameters(c_pipeline, candidate_config)
        for fold_id in range(fold_set.n_splits):
            fold_summary = fold_set.summary.iloc[fold_id]
            c_rows.append({
                "candidate_id": candidate_id,
                "fold_id": fold_id,
                "train_rows": int(fold_summary["train_rows"]),
                "train_groups": int(fold_summary["train_groups"]),
                "validation_rows": int(fold_summary["validation_rows"]),
                "validation_groups": int(fold_summary["validation_groups"]),
                "fold_train_rows_expected": int(fold_summary["train_rows"]),
                "fold_validation_rows_expected": int(fold_summary["validation_rows"]),
                "n_features": int(baseline_features[fold_id]),
                "macro_f1": float(c_score[f"fold_{fold_id}_macro_f1"]),
                "accuracy": float(c_score[f"fold_{fold_id}_accuracy"]),
                "convergence_status": "converged_no_warning",
                "warnings": " | ".join(c_search.warnings),
            })

    cv_result_rows = [row for result in candidate_results for row in _fold_rows(result, fold_set)]
    cv_results = pd.DataFrame([*cv_result_rows, *c_rows])
    cv_summary = _summary_from_results(cv_results)
    selected_candidate_id, tied_candidates = select_cv_winner(cv_summary)
    selected_mean = float(cv_summary.loc[
        cv_summary["candidate_id"].eq(selected_candidate_id), "mean_macro_f1",
    ].iloc[0])
    selection = {
        "candidate_id": selected_candidate_id,
        "selection_metric": "mean_macro_f1",
        "selected_mean_cv_macro_f1": selected_mean,
        "tie_candidates_at_max": list(tied_candidates),
        "tie_order": list(TIE_ORDER),
        "rule": "max mean CV macro F1; exact ties use SPEC order with baseline first",
        "rationale": (
            "Selected by the fixed train-only CV rule before inspecting the new validation score."
        ),
        "parameters": candidate_parameter_sets[selected_candidate_id],
        "outer_validation_used_for_selection": False,
        "held_out_test_used": False,
    }

    selected_pipeline, selected_config = _candidate_pipeline(selected_candidate_id, baseline_config)
    fit_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            selected_pipeline.fit(
                train[selected_config.text_col].tolist(),
                train[selected_config.target_col].to_numpy(dtype=np.int64),
            )
        except ConvergenceWarning as error:
            raise RuntimeError(
                f"Selected candidate {selected_candidate_id!r} raised ConvergenceWarning on full train"
            ) from error
        except Exception as error:
            raise RuntimeError(f"Selected candidate {selected_candidate_id!r} full-train fit failed: {error}") from error
        predictions = selected_pipeline.predict(validation[selected_config.text_col].tolist())
    fit_warnings.extend(f"{item.category.__name__}: {item.message}" for item in recorded_warnings)
    prediction_values = np.asarray(predictions, dtype=np.int64)
    if prediction_values.shape != (len(validation),):
        raise RuntimeError("Selected candidate validation predictions have an unexpected shape")
    if set(np.unique(prediction_values).tolist()).difference({0, 1, 2}):
        raise RuntimeError("Selected candidate produced an unknown class ID")
    selected_metrics = evaluate_predictions(
        validation[selected_config.target_col].to_numpy(dtype=np.int64), prediction_values,
    )
    reference = day04_inputs.validation_predictions
    if reference["source_row_id"].tolist() != validation["source_row_id"].tolist():
        raise RuntimeError("Day03 baseline reference is not aligned to Day04 validation rows")
    baseline_metrics = evaluate_predictions(
        reference["true_label_id"].to_numpy(dtype=np.int64),
        reference["baseline_pred_label_id"].to_numpy(dtype=np.int64),
    )
    validation_predictions = pd.DataFrame({
        "source_row_id": validation["source_row_id"].to_numpy(copy=True),
        "true_label_id": validation[selected_config.target_col].to_numpy(copy=True),
        "selected_pred_label_id": prediction_values,
        "day03_baseline_pred_label_id": reference["baseline_pred_label_id"].to_numpy(copy=True),
    })
    if validation_predictions["source_row_id"].tolist() != reference["source_row_id"].tolist():
        raise RuntimeError("Day04 validation predictions lost Day03 source row alignment")
    metrics = _metrics_payload(selected_candidate_id, selected_metrics, baseline_metrics)
    metrics["confirmation_warnings"] = fit_warnings
    artifact_paths = _artifact_paths(project_root)

    fold_manifest = fold_set.manifest.reset_index(drop=True)
    fold_summary = fold_set.summary.reset_index()
    effective_config: dict[str, object] = {
        "policy": "day04-effective-config-v1",
        "experiment_config": experiment_config,
        "baseline_config": {
            "policy": baseline_config.policy,
            "tfidf": {**baseline_config.tfidf, "ngram_range": list(baseline_config.tfidf["ngram_range"])},
            "logreg": dict(baseline_config.logreg),
        },
        "folds": {"n_splits": fold_set.n_splits, "random_state": fold_set.random_state},
        "selected_candidate_id": selected_candidate_id,
    }
    selection_bytes = _json_bytes(selection)
    config_bytes = _json_bytes(effective_config)
    fold_manifest_bytes = fold_manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    fold_summary_bytes = fold_summary.to_csv(index=False, lineterminator="\n").encode("utf-8")
    cv_results_bytes = cv_results.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    cv_summary_bytes = cv_summary.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    prediction_bytes = validation_predictions.to_csv(index=False, lineterminator="\n").encode("utf-8")
    metrics_bytes = _json_bytes(metrics)

    manifest_roundtrip = _roundtrip_csv(
        fold_manifest_bytes, int_columns=("source_row_id", "group_id", "fold_id"),
        columns=("source_row_id", "group_id", "fold_id"),
    )
    pd.testing.assert_frame_equal(manifest_roundtrip, fold_manifest, check_dtype=True)
    predictions_roundtrip = _roundtrip_csv(
        prediction_bytes,
        int_columns=("source_row_id", "true_label_id", "selected_pred_label_id", "day03_baseline_pred_label_id"),
        columns=("source_row_id", "true_label_id", "selected_pred_label_id", "day03_baseline_pred_label_id"),
    )
    pd.testing.assert_frame_equal(predictions_roundtrip, validation_predictions, check_dtype=True)
    fold_summary_roundtrip = _roundtrip_csv(
        fold_summary_bytes,
        int_columns=tuple(column for column in fold_summary.columns if column != "fold_id") + ("fold_id",),
        columns=tuple(fold_summary.columns),
    )
    pd.testing.assert_frame_equal(fold_summary_roundtrip, fold_summary, check_dtype=True)
    cv_integer_columns = (
        "fold_id", "train_rows", "train_groups", "validation_rows", "validation_groups",
        "n_features", "fold_train_rows_expected", "fold_validation_rows_expected",
    )
    cv_results_roundtrip = _roundtrip_csv(
        cv_results_bytes, int_columns=cv_integer_columns, columns=tuple(cv_results.columns),
    )
    pd.testing.assert_frame_equal(cv_results_roundtrip, cv_results, check_dtype=True)
    cv_summary_roundtrip = _roundtrip_csv(
        cv_summary_bytes, int_columns=("n_features_min", "n_features_max"),
        columns=tuple(cv_summary.columns),
    )
    pd.testing.assert_frame_equal(cv_summary_roundtrip, cv_summary, check_dtype=True)
    roundtrip_metrics = _metrics_payload(
        selected_candidate_id,
        evaluate_predictions(
            predictions_roundtrip["true_label_id"].to_numpy(dtype=np.int64),
            predictions_roundtrip["selected_pred_label_id"].to_numpy(dtype=np.int64),
        ),
        evaluate_predictions(
            predictions_roundtrip["true_label_id"].to_numpy(dtype=np.int64),
            predictions_roundtrip["day03_baseline_pred_label_id"].to_numpy(dtype=np.int64),
        ),
    )
    roundtrip_metrics["confirmation_warnings"] = fit_warnings
    if roundtrip_metrics != metrics:
        raise ValueError("Round-tripped Day04 validation predictions do not reproduce metrics")
    if json.loads(selection_bytes) != selection or json.loads(config_bytes) != effective_config:
        raise ValueError("Day04 selection/effective config failed JSON round-trip")
    if json.loads(metrics_bytes) != metrics:
        raise ValueError("Day04 metrics failed JSON round-trip")

    versions = {
        "python": platform.python_version(), "pandas": version("pandas"), "numpy": version("numpy"),
        "scipy": version("scipy"), "scikit-learn": version("scikit-learn"),
    }
    report_payloads = {
        artifact_paths.fold_manifest: fold_manifest_bytes,
        artifact_paths.fold_summary: fold_summary_bytes,
        artifact_paths.cv_results: cv_results_bytes,
        artifact_paths.cv_summary: cv_summary_bytes,
        artifact_paths.effective_config: config_bytes,
        artifact_paths.selection: selection_bytes,
        artifact_paths.validation_predictions: prediction_bytes,
        artifact_paths.metrics: metrics_bytes,
    }
    if set(report_payloads).intersection(input_bytes):
        raise ValueError("Day04 output paths must not overwrite inputs")
    changed_inputs = [path for path, content in input_bytes.items() if path.read_bytes() != content]
    if changed_inputs:
        raise ValueError(f"Day04 input bytes changed during the run; outputs not written: {changed_inputs}")
    for path, content in report_payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    provenance = {
        "policy": "day04-provenance-v1",
        "source": {"day02_provenance_sha256": _sha256(input_bytes[config_source_path])},
        "day03_provenance_sha256": _sha256(input_bytes[day03_report / "provenance.json"]),
        "input_checksums": {
            str(path.relative_to(project_root)): _sha256(content) for path, content in input_bytes.items()
        },
        "output_checksums": {
            path.name: _sha256(content) for path, content in report_payloads.items()
        },
        "cv": {"n_splits": fold_set.n_splits, "random_state": fold_set.random_state},
        "selection": {"candidate_id": selected_candidate_id, "metric": "mean_macro_f1"},
        "versions": versions,
        "class_order": list(CLASS_ORDER),
    }
    artifact_paths.provenance.parent.mkdir(parents=True, exist_ok=True)
    artifact_paths.provenance.write_bytes(_json_bytes(provenance))
    return Day04Run(
        fold_set=fold_set, cv_results=cv_results, cv_summary=cv_summary,
        selection=selection, validation_predictions=validation_predictions,
        metrics=metrics, artifacts=artifact_paths,
    )


def main() -> None:
    """Run the supported offline Day04 experiment CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare the fixed Day04 candidate set offline.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--config", type=Path, default=EXPERIMENT_CONFIG)
    args = parser.parse_args()
    result = run_experiments_day04(args.config, args.root)
    print(f"Selected candidate: {result.selection['candidate_id']}")
    print(f"CV summary: {result.artifacts.cv_summary}")
    print(f"Validation metrics: {result.artifacts.metrics}")


if __name__ == "__main__":
    main()
