"""Consistency evidence is published only after frozen and isolated checks pass."""

from dataclasses import replace
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.features import preprocess
from finnews_sentiment.models import inference_artifact as artifact
from finnews_sentiment.models import inference_consistency as consistency
from finnews_sentiment.models import predict
from test_day06_artifact import saved_pair, completed_reference, day04_reference


def _forbid(*args: object, **kwargs: object) -> None:
    raise AssertionError("Consistency must not fit, serialize, run CV/final evaluation or access network")


@pytest.fixture
def root(completed_reference: Path, saved_pair: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    shutil.copytree(completed_reference, root)
    shutil.copytree(saved_pair.parent, root / "models/day06")
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "fit_transform", _forbid)
    monkeypatch.setattr("joblib.dump", _forbid)
    monkeypatch.setattr(artifact, "build_inference_bundle", _forbid)
    monkeypatch.setattr("finnews_sentiment.models.final_evaluation.run_final_evaluation", _forbid)
    monkeypatch.setattr("finnews_sentiment.models.cv_runner.run_cv_candidate", _forbid)
    monkeypatch.setattr("socket.create_connection", _forbid)
    return root


def _files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_complete_report_repeat_and_isolated_process(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = _files(root)
    original = subprocess.run
    launches = []

    def inspect(command, **kwargs):
        workspace = Path(kwargs["cwd"])
        assert all(not (workspace / name).exists() for name in ("data", "configs", "reports"))
        assert (workspace / "src/finnews_sentiment").is_dir()
        assert set(path.name for path in (workspace / "artifact").iterdir()) == {artifact.BUNDLE_NAME, artifact.MANIFEST_NAME}
        assert command[1:3] == ["-I", "-B"]
        launches.append("isolated")
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", inspect)
    first = consistency.check_inference_consistency(root)
    content = first.report_path.read_bytes()
    mtime = first.report_path.stat().st_mtime_ns
    second = consistency.check_inference_consistency(root)
    assert launches == ["isolated", "isolated"]
    assert second.report_path.read_bytes() == content
    assert second.report_path.stat().st_mtime_ns == mtime
    assert first.report == second.report == json.loads(content)
    parity = first.report["parity"]
    assert parity["reference_sha256"] == parity["pipeline_sha256"] == parity["public_api_sha256"]
    assert parity["n_train"] > 0 and parity["n_validation"] > 0
    assert first.report["state_before_sha256"] == first.report["state_after_sha256"]
    assert first.report["serialization_evidence"].startswith("D6-03")
    assert first.report["standalone"]["status"] == "passed"
    assert first.report["standalone"]["file_guard_checked"]
    assert first.report["standalone"]["import_without_artifact_io"]
    assert all((root / name).read_bytes() == content for name, content in before.items())
    assert set(_files(root)) - set(before) == {"reports/day06/consistency.json"}


@pytest.mark.parametrize("split", ["train", "validation"])
def test_raw_clean_mismatch_is_not_reported_as_passed(root: Path, monkeypatch: pytest.MonkeyPatch, split: str) -> None:
    reference = artifact.load_inference_reference(root)
    target = str(getattr(reference, split)["text"].iloc[0])
    original = preprocess.preprocess_text
    monkeypatch.setattr(preprocess, "preprocess_text", lambda text: "changed" if text == target else original(text))
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    with pytest.raises(ValueError, match=f"Raw/clean mismatch: {split}"):
        consistency.check_inference_consistency(root)
    assert not (root / "reports/day06").exists()


@pytest.mark.parametrize("damage", ["ids", "truth", "prediction"])
def test_reference_alignment_and_prediction_mismatch(root: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    original = artifact.load_inference_reference

    def changed(project_root: Path):
        reference = original(project_root)
        predictions = reference.validation_predictions.copy()
        if damage == "ids":
            predictions = predictions.iloc[::-1].copy()
        elif damage == "truth":
            predictions.loc[predictions.index[0], "true_label_id"] = 99
        else:
            predictions["pred_label_id"] = (predictions["pred_label_id"] + 1) % 3
        return replace(reference, validation_predictions=predictions)

    monkeypatch.setattr(artifact, "load_inference_reference", changed)
    with pytest.raises(ValueError, match="IDs/order|truth|prediction mismatch"):
        consistency.check_inference_consistency(root)
    assert not (root / "reports/day06").exists()


@pytest.mark.parametrize("damage", ["raw", "clean", "label", "name"])
def test_public_api_mismatch_is_rejected(root: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    original = predict.predict_texts

    def altered(bundle, texts):
        results = original(bundle, texts)
        first = results[0]
        if damage == "raw":
            results[0] = replace(first, text="altered raw")
        elif damage == "clean":
            results[0] = replace(first, text_clean="altered clean")
        elif damage == "name":
            results[0] = replace(first, pred_label="wrong name")
        else:
            label = (first.pred_label_id + 1) % 3
            results[0] = replace(first, pred_label_id=label, pred_label=("negative", "neutral", "positive")[label])
        return results

    monkeypatch.setattr(predict, "predict_texts", altered)
    with pytest.raises(ValueError, match="API|prediction mismatch"):
        consistency.check_inference_consistency(root)
    assert not (root / "reports/day06").exists()


def test_predict_state_mutation_blocks_report(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Pipeline.predict

    def mutate(self: Pipeline, texts: list[str]) -> np.ndarray:
        result = np.asarray(original(self, texts))
        self.named_steps["clf"].coef_[0, 0] += 1
        return result

    monkeypatch.setattr(Pipeline, "predict", mutate)
    with pytest.raises(ValueError, match="state/metadata mutation"):
        consistency.check_inference_consistency(root)
    assert not (root / "reports/day06").exists()


@pytest.mark.parametrize("name", ["models/day06/sentiment_bundle.joblib", "reports/day04/validation_predictions.csv"])
def test_altered_artifact_or_reference_preserves_previous_report(root: Path, name: str) -> None:
    report = root / "reports/day06/consistency.json"
    report.parent.mkdir()
    report.write_bytes(b"previous result")
    path = root / name
    path.write_bytes(path.read_bytes() + b"corrupt")
    before = _files(root)
    with pytest.raises(ValueError, match="checksum mismatch"):
        consistency.check_inference_consistency(root)
    assert before == _files(root)


@pytest.mark.parametrize("failure", ["exit", "json", "evidence", "timeout"])
def test_child_failure_does_not_publish(root: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    def failed(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 60)
        return subprocess.CompletedProcess(command, 1 if failure == "exit" else 0,
                                           stdout="not json" if failure == "json" else "{}", stderr="fixture child failure")

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(ValueError, match="Standalone|standalone"):
        consistency.check_inference_consistency(root)
    assert not (root / "reports/day06").exists()


@pytest.mark.parametrize("impurity", ["artifact_io", "fit", "network"])
def test_child_catches_import_side_effects(saved_pair: Path, tmp_path: Path, impurity: str) -> None:
    source = tmp_path / "src"
    shutil.copytree(ROOT / "src/finnews_sentiment", source / "finnews_sentiment", ignore=shutil.ignore_patterns("__pycache__"))
    snippets = {
        "artifact_io": "\nfrom pathlib import Path as _ProbePath\n(_ProbePath(__file__).resolve().parents[3] / 'artifact/sentiment_bundle.joblib').read_bytes()\n",
        "fit": "\nfrom sklearn.pipeline import Pipeline as _ProbePipeline\n_ProbePipeline([]).fit([], [])\n",
        "network": "\nimport socket as _ProbeSocket\n_ProbeSocket.create_connection(('example.invalid', 80))\n",
    }
    path = source / "finnews_sentiment/models/predict.py"
    path.write_text(path.read_text() + snippets[impurity])
    # A failing child must be rejected before any expected-evidence comparison.
    expected: consistency.StandaloneEvidence = {
        "status": "passed", "import_without_artifact_io": True,
        "training_guard_checked": True, "file_guard_checked": True, "network_guard_checked": True,
        "state_sha256": "unreachable", "runtime_versions": {}, "predictions": [],
    }
    with pytest.raises(ValueError, match="Standalone inference failed"):
        consistency._check_standalone(saved_pair, expected, source)


def test_import_has_no_data_model_io_or_process_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", _forbid)
        context.setattr(Path, "read_text", _forbid)
        context.setattr(Pipeline, "fit", _forbid)
        context.setattr(subprocess, "run", _forbid)
        importlib.reload(consistency)
