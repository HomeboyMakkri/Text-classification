"""Offline report consumes references/cache; preserves inputs and repeats exactly."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.data.experiment_splits import load_day04_inputs
from finnews_sentiment.models import final_evaluation as final
from finnews_sentiment.models import run_error_analysis_day05 as report
from finnews_sentiment.models.evaluate import evaluate_predictions
from finnews_sentiment.models.failure_cases import ANNOTATION_COLUMNS, select_validation_examples
from finnews_sentiment.models.error_analysis import load_validation_analysis
from finnews_sentiment.models.run_experiments_day04 import _metrics_payload
from test_day05_final_evaluation import root, day04_reference


def _files(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _forbid(*args: object, **kwargs: object) -> None:
    raise AssertionError("Report must not fit, predict, train CV or access network")


@pytest.fixture(params=[False, True], ids=["no-errors", "reviewed-errors"])
def report_root(root: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Path:
    if request.param:
        inputs = load_day04_inputs(root)
        directory = root / "reports/day04"
        predictions = pd.read_csv(directory / "validation_predictions.csv")
        truth = predictions["true_label_id"].to_numpy()
        predicted = (truth + 1 + np.arange(len(truth)) % 2) % 3
        predictions["selected_pred_label_id"] = predicted
        predictions.to_csv(directory / "validation_predictions.csv", index=False)
        metrics = _metrics_payload("baseline", evaluate_predictions(truth, predicted), evaluate_predictions(
            truth, predictions["day03_baseline_pred_label_id"].to_numpy(),
        ))
        metrics["confirmation_warnings"] = []
        (directory / "metrics.json").write_text(json.dumps(metrics))
        provenance_path = directory / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        for name in ("validation_predictions.csv", "metrics.json"):
            provenance["output_checksums"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        provenance_path.write_text(json.dumps(provenance))
        validation_texts = inputs.splits["validation"]["text_clean"].tolist()

        def fixture_predict(self: Pipeline, X: list[str]) -> np.ndarray:
            return predicted.copy() if X == validation_texts else np.ones(len(X), dtype=np.int64)

        # Fixture-only reference predictions; no real data or final cache involved.
        with monkeypatch.context() as context:
            context.setattr(Pipeline, "predict", fixture_predict)
            final.run_final_evaluation(root)
    else:
        final.run_final_evaluation(root)
    examples = select_validation_examples(load_validation_analysis(root).table).examples
    annotations = pd.DataFrame({
        "source_row_id": examples["source_row_id"].tolist(),
        "primary_category": ["other/unclear"] * len(examples),
        "observation": ['Literal "NA", comma and\nnewline observation'] * len(examples),
        "hypothesis": [""] * len(examples), "rationale": ["Fixture review; no causal claim"] * len(examples),
    }, columns=list(ANNOTATION_COLUMNS))
    (root / "docs").mkdir()
    annotations.to_csv(root / "docs/day05_failure_cases.csv", index=False)
    return root


def test_report_end_to_end_roundtrip_and_repeat(report_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    monkeypatch.setattr(final, "run_final_evaluation", _forbid)
    monkeypatch.setattr("socket.create_connection", _forbid)
    monkeypatch.setattr("finnews_sentiment.models.cv_runner.run_cv_candidate", _forbid)
    before = _files(report_root)
    result = report.run_error_analysis_day05(report_root)
    assert result.final.cache_status == "reused"
    assert before == {path: path.read_bytes() for path in before}
    assert len(result.artifacts) == 11
    assert all(path.parent == report_root / "reports/day05" for path in result.artifacts.values())
    assert not set(final.FINAL_NAMES).intersection(result.artifacts)
    assert not ({"provenance.json"} & set(result.provenance["output_checksums"]))
    assert result.summary["validation"]["n_errors"] == len(result.analysis.errors)
    assert result.summary["manual_review"]["n_reviewed_rows"] == len(result.reviewed_examples)
    assert result.summary["test"]["evaluation_split"] == "test"
    assert result.summary["test"]["macro_f1"] == result.final.confusion.metrics.macro_f1
    examples = pd.read_csv(result.artifacts["validation_examples.csv"], keep_default_na=False)
    assert examples["source_row_id"].tolist() == result.reviewed_examples["source_row_id"].tolist()
    if len(examples):
        assert examples["hypothesis"].eq("").all()
        assert examples["observation"].eq('Literal "NA", comma and\nnewline observation').all()
    summary = json.loads(result.artifacts["analysis_summary.json"].read_text())
    assert summary == result.summary
    provenance = json.loads(result.artifacts["provenance.json"].read_text())
    for name, expected in provenance["input_checksums"].items():
        assert hashlib.sha256((report_root / name).read_bytes()).hexdigest() == expected
    for name, expected in provenance["output_checksums"].items():
        assert hashlib.sha256(result.artifacts[name].read_bytes()).hexdigest() == expected
    assert provenance["final_evaluation_sha256"] == provenance["input_checksums"]["reports/day05/final_evaluation.json"]
    text = result.artifacts["observations.md"].read_text()
    assert "## Validation" in text and "## Frozen held-out test" in text
    assert "reviewed rows only" in text
    if examples.empty:
        assert "Empty pair:" in text
    frozen = _files(report_root)
    repeated = report.run_error_analysis_day05(report_root)
    assert repeated.summary == result.summary
    assert repeated.provenance == result.provenance
    assert frozen == _files(report_root)


@pytest.mark.parametrize("damage", ["missing-cache", "cache-checksum", "annotations", "input"])
def test_bad_inputs_leave_existing_report_untouched(report_root: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    report.run_error_analysis_day05(report_root)
    if damage == "missing-cache":
        (report_root / "reports/day05/final_evaluation.json").unlink()
    elif damage == "cache-checksum":
        path = report_root / "reports/day05/test_predictions.csv"
        path.write_bytes(path.read_bytes() + b"corrupt")
    elif damage == "annotations":
        path = report_root / "docs/day05_failure_cases.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        extra = pd.DataFrame([[999999, "other/unclear", "Extra ID", "", "Not selected"]], columns=list(ANNOTATION_COLUMNS))
        pd.concat([frame, extra]).to_csv(path, index=False)
    else:
        path = report_root / "configs/experiments_day04.json"
        path.write_text(path.read_text() + " ")
    before = _files(report_root)
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    with pytest.raises(ValueError):
        report.run_error_analysis_day05(report_root)
    assert before == _files(report_root)


def test_input_mutation_during_build_is_detected_before_writes(report_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = report_root / "docs/day05_failure_cases.csv"

    def mutate(*args: object, **kwargs: object) -> str:
        path.write_bytes(path.read_bytes() + b"\n")
        return "not to be written"

    monkeypatch.setattr(report, "_observations", mutate)
    directory = report_root / "reports/day05"
    before = _files(directory)
    with pytest.raises(ValueError, match="input changed"):
        report.run_error_analysis_day05(report_root)
    assert before == _files(directory)


def test_import_has_no_io_or_execution() -> None:
    code = '''
from pathlib import Path
from unittest.mock import patch
import finnews_sentiment.models.run_error_analysis_day05 as module
import importlib
with patch.object(Path, "read_bytes", side_effect=AssertionError("import I/O")), patch.object(Path, "write_bytes", side_effect=AssertionError("import write")), patch.object(module, "load_final_evaluation", side_effect=AssertionError("import cache")):
    importlib.reload(module)
'''
    result = subprocess.run([str(ROOT / "venv/bin/python"), "-c", code], cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_report_cli_repeat_without_model_calls(report_root: Path) -> None:
    code = '''
import runpy
import sys
from sklearn.pipeline import Pipeline

def forbidden(*args, **kwargs):
    raise AssertionError("Report CLI must not fit/predict")

Pipeline.fit = forbidden
Pipeline.predict = forbidden
sys.argv = ["report", "--root", sys.argv[1]]
runpy.run_module("finnews_sentiment.models.run_error_analysis_day05", run_name="__main__")
'''
    before = _files(report_root)
    for _ in range(2):
        result = subprocess.run([str(ROOT / "venv/bin/python"), "-c", code, str(report_root)], cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "final cache: reused" in result.stdout
        assert before == {path: path.read_bytes() for path in before}
        before = _files(report_root)
