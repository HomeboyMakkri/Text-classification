"""Frozen train-only replay, single test prediction and strict cache reuse."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.data.experiment_splits import load_day04_inputs
from finnews_sentiment.models import final_evaluation as final
from test_day05_error_analysis import day04_reference


@pytest.fixture
def root(day04_reference: Path, tmp_path: Path) -> Path:
    root = tmp_path / "project"
    shutil.copytree(day04_reference, root)
    path = root / "reports/day04/provenance.json"
    provenance = json.loads(path.read_text())
    versions = final.runtime_versions()
    provenance["versions"] = {name: versions[name] for name in ("python", "pandas", "numpy", "scipy", "scikit-learn")}
    path.write_text(json.dumps(provenance))
    return root


def _files(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _forbid(*args: object, **kwargs: object) -> None:
    raise AssertionError("Cache reuse must not fit/predict")


def test_train_only_freeze_replay_test_then_read_only_reuse(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = load_day04_inputs(root)
    original_fit = Pipeline.fit
    original_predict = Pipeline.predict
    calls: list[str] = []
    before = _files(root)
    train = inputs.splits["train"]
    validation = inputs.splits["validation"]
    test = inputs.splits["test"]

    def fit_spy(self: Pipeline, X: list[str], y: np.ndarray) -> Pipeline:
        protocol = json.loads((root / "reports/day05/frozen_protocol.json").read_text())
        assert protocol["fit_policy"] == "day03_train_only"
        assert protocol["pipeline_parameters"]["tfidf"]["dtype"] == "float64"
        assert protocol["pipeline_parameters"]["tfidf"]["ngram_range"] == [1, 2]
        assert protocol["pipeline_parameters"]["classifier"]["C"] == 1.0
        assert protocol["train_source_row_ids"] == train["source_row_id"].tolist()
        assert X == train["text_clean"].tolist()
        np.testing.assert_array_equal(y, train["label_id"].to_numpy())
        assert not (root / "reports/day05/final_evaluation.json").exists()
        calls.append("fit_train")
        return original_fit(self, X, y)

    def predict_spy(self: Pipeline, X: list[str]) -> np.ndarray:
        assert (root / "reports/day05/frozen_protocol.json").exists()
        if calls == ["fit_train"]:
            assert X == validation["text_clean"].tolist()
            calls.append("validation_replay")
        else:
            assert calls == ["fit_train", "validation_replay"]
            assert X == test["text_clean"].tolist()
            calls.append("test_predict")
        return np.asarray(original_predict(self, X))

    monkeypatch.setattr(Pipeline, "fit", fit_spy)
    monkeypatch.setattr(Pipeline, "predict", predict_spy)
    created = final.run_final_evaluation(root)
    assert created.cache_status == "created"
    assert calls == ["fit_train", "validation_replay", "test_predict"]
    assert created.predictions["source_row_id"].tolist() == test["source_row_id"].tolist()
    assert created.n_train == len(train)
    assert created.n_features > 0
    assert created.confusion.counts.to_numpy().sum() == len(test)
    marker = json.loads((created.directory / "final_evaluation.json").read_text())
    assert "final_evaluation.json" not in marker["output_checksums"]
    assert before == {path: path.read_bytes() for path in before}
    frozen = _files(root)
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    for result in (final.run_final_evaluation(root), final.load_final_evaluation(root)):
        assert result.cache_status == "reused"
        assert result.protocol == created.protocol
        pd.testing.assert_frame_equal(result.predictions, created.predictions)
        assert result.confusion.metrics == created.confusion.metrics
    assert frozen == _files(root)


@pytest.mark.parametrize("failure", ["replay", "fit_warning", "replay_warning"])
def test_failure_stops_before_test_and_blocks_retry(root: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    original_fit = Pipeline.fit
    original_predict = Pipeline.predict
    calls: list[str] = []

    def fit_spy(self: Pipeline, X: list[str], y: np.ndarray) -> Pipeline:
        calls.append("fit")
        if failure == "fit_warning":
            warnings.warn("fixture convergence", ConvergenceWarning)
        return original_fit(self, X, y)

    def predict_spy(self: Pipeline, X: list[str]) -> np.ndarray:
        assert calls == ["fit"]  # a second prediction would be test leakage
        calls.append("validation")
        if failure == "replay_warning":
            warnings.warn("fixture replay convergence", ConvergenceWarning)
        predicted = np.asarray(original_predict(self, X))
        if failure == "replay":
            predicted[0] = (predicted[0] + 1) % 3
        return predicted

    monkeypatch.setattr(Pipeline, "fit", fit_spy)
    monkeypatch.setattr(Pipeline, "predict", predict_spy)
    with pytest.raises(ValueError, match="replay mismatch" if failure == "replay" else "fixture"):
        final.run_final_evaluation(root)
    assert (root / "reports/day05/frozen_protocol.json").exists()
    assert not (root / "reports/day05/test_predictions.csv").exists()
    partial = _files(root)
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    with pytest.raises(ValueError, match="Incomplete final cache"):
        final.run_final_evaluation(root)
    assert partial == _files(root)


@pytest.mark.parametrize("damage", ["protocol", "checksum", "missing", "marker", "truth", "ids", "predictions", "metrics", "matrix", "runtime", "replay"])
def test_cache_corruption_never_retrains(root: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    created = final.run_final_evaluation(root)
    directory = created.directory
    marker_path = directory / "final_evaluation.json"
    marker = json.loads(marker_path.read_text())
    changed_output: Path | None = None
    if damage == "protocol":
        path = directory / "frozen_protocol.json"
        payload = json.loads(path.read_text())
        payload["random_state"] = 99
        path.write_text(json.dumps(payload))
    elif damage == "checksum":
        path = directory / "test_confusion_counts.png"
        path.write_bytes(path.read_bytes() + b"corrupt")
    elif damage == "missing":
        (directory / "test_metrics.json").unlink()
    elif damage == "marker":
        marker["protocol_sha256"] = "0" * 64
    elif damage in ("truth", "ids", "predictions"):
        path = directory / "test_predictions.csv"
        frame = pd.read_csv(path)
        if damage == "truth":
            frame.loc[0, "true_label_id"] = (int(frame["true_label_id"].to_numpy(dtype="int64")[0]) + 1) % 3
        elif damage == "ids":
            frame = frame.iloc[::-1]
        else:
            frame.loc[0, "pred_label_id"] = 7
        frame.to_csv(path, index=False)
        changed_output = path
    elif damage == "metrics":
        path = directory / "test_metrics.json"
        payload = json.loads(path.read_text())
        payload["macro_f1"] = -1
        path.write_text(json.dumps(payload))
        changed_output = path
    elif damage == "matrix":
        path = directory / "test_confusion_counts.csv"
        frame = pd.read_csv(path, index_col=0)
        frame.iloc[0, 0] = int(frame.to_numpy(dtype="int64")[0, 0]) + 1
        frame.to_csv(path)
        changed_output = path
    elif damage == "runtime":
        marker["runtime_versions"]["numpy"] = "different"
    else:
        marker["validation_replay"]["status"] = "not_checked"
    if changed_output is not None:
        marker["output_checksums"][changed_output.name] = hashlib.sha256(changed_output.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker))
    before = _files(root)
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    for reader in (final.run_final_evaluation, final.load_final_evaluation):
        with pytest.raises(ValueError, match="Final evaluation stopped"):
            reader(root)
    assert before == _files(root)


@pytest.mark.parametrize("failure", ["version", "input", "config"])
def test_invalid_reference_stops_before_freeze(root: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    if failure == "version":
        versions = final.runtime_versions()
        monkeypatch.setattr(final, "runtime_versions", lambda: {**versions, "numpy": "different"})
    elif failure == "input":
        path = root / "data/processed/financial_phrasebank_75agree.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path = root / "configs/train_baseline.json"
        path.write_text(path.read_text() + " ")
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    with pytest.raises(ValueError, match="mismatch"):
        final.run_final_evaluation(root)
    assert not (root / "reports/day05/frozen_protocol.json").exists()


def test_read_only_missing_cache_has_no_side_effects(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    before = _files(root)
    with pytest.raises(ValueError, match="Run explicitly"):
        final.load_final_evaluation(root)
    assert before == _files(root)
