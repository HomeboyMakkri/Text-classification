"""Validation joins reject corrupt references and preserve source text."""

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.models.error_analysis import build_validation_table, load_validation_analysis
from finnews_sentiment.models.run_experiments_day04 import (
    TIE_ORDER, _candidate_pipeline, _candidate_parameters, _expected_experiment_config, _metrics_payload,
)
from finnews_sentiment.data.baseline_config import load_baseline_config
from finnews_sentiment.data.experiment_splits import load_day04_inputs
from finnews_sentiment.models.evaluate import evaluate_predictions
from test_day04_splits import _make_reference


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation = pd.DataFrame({"source_row_id": [4, 8], "text": ["Raw A", "Raw B"], "text_clean": ["a", "b"], "label_id": [0, 2]})
    manifest = pd.DataFrame({"source_row_id": [1, 4, 8, 9], "group_id": [1, 4, 8, 9], "split": ["train", "validation", "validation", "test"]})
    predictions = pd.DataFrame({"source_row_id": [4, 8], "true_label_id": [0, 2], "selected_pred_label_id": [1, 2], "day03_baseline_pred_label_id": [0, 2]})
    return validation, manifest, predictions


def test_join_and_no_errors(frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    validation, manifest, predictions = frames
    validation.index = [88, 77]
    before = validation.copy(deep=True)
    table = build_validation_table(validation, manifest, predictions)
    assert table["text"].tolist() == ["Raw A", "Raw B"]
    assert table["is_error"].tolist() == [True, False]
    assert table["pred_label"].tolist() == ["neutral", "positive"]
    pd.testing.assert_frame_equal(validation, before)
    predictions["selected_pred_label_id"] = predictions["true_label_id"]
    assert not build_validation_table(validation, manifest, predictions)["is_error"].any()


@pytest.mark.parametrize("damage", ["reorder", "missing", "extra", "duplicate", "truth", "unknown", "null", "schema"])
def test_rejects_predictions(frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], damage: str) -> None:
    validation, manifest, predictions = frames
    if damage == "reorder":
        predictions = predictions.iloc[::-1]
    elif damage == "missing":
        predictions = predictions.iloc[:1]
    elif damage == "extra":
        predictions = pd.concat([predictions, predictions.iloc[:1].assign(source_row_id=99)])
    elif damage == "duplicate":
        predictions.loc[1, "source_row_id"] = 4
    elif damage == "truth":
        predictions.loc[0, "true_label_id"] = 2
    elif damage == "unknown":
        predictions.loc[0, "selected_pred_label_id"] = 3
    elif damage == "null":
        predictions = predictions.astype("Int64")
        predictions.loc[0, "selected_pred_label_id"] = pd.NA
    else:
        predictions = predictions.drop(columns="true_label_id")
    with pytest.raises(ValueError):
        build_validation_table(validation, manifest, predictions)


@pytest.mark.parametrize("damage", ["empty", "blank", "null", "overlap"])
def test_rejects_inputs(frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], damage: str) -> None:
    validation, manifest, predictions = frames
    if damage == "empty":
        validation = validation.iloc[:0]
    elif damage == "blank":
        validation.loc[0, "text_clean"] = " "
    elif damage == "null":
        validation.loc[0, "text"] = None
    else:
        manifest.loc[0, "group_id"] = 4
    with pytest.raises(ValueError):
        build_validation_table(validation, manifest, predictions)


@pytest.fixture(scope="module")
def day04_reference(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = _make_reference(tmp_path_factory.mktemp("day05"))
    shutil.copyfile(ROOT / "configs/experiments_day04.json", root / "configs/experiments_day04.json")
    config = load_baseline_config(root / "configs/train_baseline.json")
    inputs = load_day04_inputs(root)
    reference = inputs.validation_predictions
    candidate = "baseline"
    pipeline, candidate_config = _candidate_pipeline(candidate, config)
    parameters = _candidate_parameters(pipeline, candidate_config)
    directory = root / "reports/day04"
    directory.mkdir(parents=True)
    selection = {"candidate_id": candidate, "parameters": parameters,
                 "selection_metric": "mean_macro_f1", "tie_order": list(TIE_ORDER), "selected_mean_cv_macro_f1": 0.5,
                 "tie_candidates_at_max": list(TIE_ORDER), "held_out_test_used": False,
                 "outer_validation_used_for_selection": False}
    effective = {"policy": "day04-effective-config-v1", "experiment_config": _expected_experiment_config(),
                 "baseline_config": {"policy": config.policy, "tfidf": {**config.tfidf, "ngram_range": list(config.tfidf["ngram_range"])}, "logreg": dict(config.logreg)},
                 "folds": {"n_splits": 3, "random_state": 42}, "selected_candidate_id": candidate}
    predictions = reference.rename(columns={"baseline_pred_label_id": "selected_pred_label_id"}).drop(columns="dummy_pred_label_id")
    predictions["day03_baseline_pred_label_id"] = predictions["selected_pred_label_id"]
    metric = evaluate_predictions(predictions["true_label_id"].to_numpy(), predictions["selected_pred_label_id"].to_numpy())
    metrics = _metrics_payload(candidate, metric, metric)
    metrics["confirmation_warnings"] = []
    for name, payload in (("selection.json", selection), ("effective_config.json", effective), ("metrics.json", metrics)):
        (directory / name).write_text(json.dumps(payload))
    predictions.to_csv(directory / "validation_predictions.csv", index=False)
    pd.DataFrame({"candidate_id": TIE_ORDER, "mean_macro_f1": [0.5] * len(TIE_ORDER)}).to_csv(directory / "cv_summary.csv", index=False)
    for name in ("cv_folds.csv", "cv_fold_summary.csv", "cv_results.csv"):
        (directory / name).write_text("fixture\n")
    input_paths = [*sorted((root / "configs").glob("*.json")), root / config.input_path,
                   root / config.provenance_path, root / config.split_output_dir / "splits.csv",
                   *(root / config.report_output_dir / name for name in ("split_summary.json", "validation_predictions.csv", "metrics.json", "config.json", "provenance.json"))]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    provenance = {"cv": {"n_splits": 3, "random_state": 42}, "policy": "day04-provenance-v1", "class_order": ["negative", "neutral", "positive"],
                  "input_checksums": hashes,
                  "output_checksums": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()},
                  "source": {"day02_provenance_sha256": hashes[str(config.provenance_path)]},
                  "day03_provenance_sha256": hashes[str(config.report_output_dir / "provenance.json")],
                  "selection": {"candidate_id": candidate, "metric": "mean_macro_f1"}}
    (directory / "provenance.json").write_text(json.dumps(provenance))
    return root


def test_loader_read_only(day04_reference: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sklearn.pipeline import Pipeline

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Analysis must not fit/predict")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(Pipeline, "predict", forbidden)
    paths = [path for path in day04_reference.rglob("*") if path.is_file()]
    before = {path: path.read_bytes() for path in paths}
    result = load_validation_analysis(day04_reference)
    assert len(result.table) > 0
    assert result.errors.equals(result.table.loc[result.table["is_error"]])
    assert before == {path: path.read_bytes() for path in paths}


@pytest.mark.parametrize("rehash", [False, True])
def test_loader_corruption(day04_reference: Path, tmp_path: Path, rehash: bool) -> None:
    root = tmp_path / "copy"
    shutil.copytree(day04_reference, root)
    path = root / "reports/day04/validation_predictions.csv"
    predictions = pd.read_csv(path).iloc[::-1]
    predictions.to_csv(path, index=False)
    if rehash:
        provenance_path = root / "reports/day04/provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["output_checksums"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="manifest order" if rehash else "checksum mismatch"):
        load_validation_analysis(root)
