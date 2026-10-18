"""Day06 preparation verifies frozen references without fitting or predicting."""

import hashlib
import importlib
import json
from pathlib import Path
import shutil
import socket
import sys
import urllib.request

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.models import final_evaluation as final
from finnews_sentiment.models import inference_artifact as artifact
from test_day05_error_analysis import day04_reference


@pytest.fixture(scope="module")
def completed_reference(day04_reference: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create an offline fixture once; all preparation tests forbid model calls."""
    root = tmp_path_factory.mktemp("day06") / "project"
    shutil.copytree(day04_reference, root)
    path = root / "reports/day04/provenance.json"
    provenance = json.loads(path.read_text())
    versions = final.runtime_versions()
    provenance["versions"] = {name: versions[name] for name in (
        "python", "pandas", "numpy", "scipy", "scikit-learn",
    )}
    path.write_text(json.dumps(provenance))
    final.run_final_evaluation(root)  # Synthetic fixture setup only, never real artifacts.
    return root


def _forbid(*args: object, **kwargs: object) -> None:
    raise AssertionError("D6-01 must not fit, predict, run CV/final evaluation or use network")


@pytest.fixture
def frozen_root(completed_reference: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    shutil.copytree(completed_reference, root)
    for name in ("fit", "fit_transform", "predict", "transform", "score"):
        monkeypatch.setattr(Pipeline, name, _forbid)
    for name in ("fit", "fit_transform", "transform"):
        monkeypatch.setattr(TfidfVectorizer, name, _forbid)
    monkeypatch.setattr(final, "run_final_evaluation", _forbid)
    monkeypatch.setattr("finnews_sentiment.models.cv_runner.run_cv_candidate", _forbid)
    monkeypatch.setattr(socket, "create_connection", _forbid)
    monkeypatch.setattr(urllib.request, "urlopen", _forbid)
    return root


def _files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _write_protocol(root: Path, payload: dict[str, object]) -> None:
    path = root / "reports/day05/frozen_protocol.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    marker_path = root / "reports/day05/final_evaluation.json"
    marker = json.loads(marker_path.read_text())
    marker["protocol_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker))


def test_read_only_reference_exposes_only_train_validation(frozen_root: Path) -> None:
    before = _files(frozen_root)
    reference = artifact.load_inference_reference(frozen_root)
    again = artifact.load_inference_reference(frozen_root)
    contract = reference.contract
    protocol = json.loads((frozen_root / "reports/day05/frozen_protocol.json").read_text())
    assert contract.candidate_id == "baseline"  # Dynamic selection, not hardcoded real winner.
    assert contract.schema_version == "day06-inference-v1"
    assert contract.fit_policy == "day03_train_only"
    assert contract.preprocessing_policy == "minimal-v1"
    assert contract.label_mapping == {"negative": 0, "neutral": 1, "positive": 2}
    assert contract.class_order == ("negative", "neutral", "positive")
    assert contract.pipeline_parameters == protocol["pipeline_parameters"]
    assert contract.pipeline_parameters["tfidf"]["dtype"] == "float64"
    assert contract.runtime_versions == final.runtime_versions()
    assert contract.preprocessing_sha256 == hashlib.sha256(
        (ROOT / "src/finnews_sentiment/features/preprocess.py").read_bytes(),
    ).hexdigest()
    assert contract.train_source_row_ids == tuple(protocol["train_source_row_ids"])
    assert contract.validation_source_row_ids == tuple(protocol["validation_source_row_ids"])
    assert reference.validation_predictions["source_row_id"].tolist() == reference.validation["source_row_id"].tolist()
    assert reference.validation_predictions["true_label_id"].tolist() == reference.validation["label_id"].tolist()
    assert not set(reference.train["text_clean"]).intersection(reference.validation["text_clean"])
    assert reference.day05_validation_replay == "exact_match"
    assert reference.day05_n_features > 0
    assert not hasattr(reference, "pipeline") and not hasattr(reference, "test")
    assert not set(protocol["test_source_row_ids"]).intersection(contract.train_source_row_ids)
    assert not set(protocol["test_source_row_ids"]).intersection(contract.validation_source_row_ids)
    assert set(protocol["input_checksums"]).issubset(contract.input_checksums)
    assert {f"reports/day05/{name}" for name in final.FINAL_NAMES}.issubset(contract.input_checksums)
    for name, expected in contract.input_checksums.items():
        assert hashlib.sha256((frozen_root / name).read_bytes()).hexdigest() == expected
    pd.testing.assert_frame_equal(reference.train, again.train)
    pd.testing.assert_frame_equal(reference.validation, again.validation)
    assert contract == again.contract
    reference.train.loc[reference.train.index[0], "text"] = "caller mutation"
    reference.contract.label_mapping["negative"] = 999
    assert again.contract.label_mapping["negative"] == 0
    assert again.train["text"].iloc[0] != "caller mutation"
    assert before == _files(frozen_root)


@pytest.mark.parametrize("name", [
    "data/processed/financial_phrasebank_75agree.csv",
    "data/processed/day03/splits.csv",
    "reports/day02/provenance.json",
    "reports/day04/selection.json",
    "reports/day05/frozen_protocol.json",
    "reports/day05/final_evaluation.json",
    "reports/day05/test_predictions.csv",
])
def test_missing_or_partial_reference_is_not_rebuilt(frozen_root: Path, name: str) -> None:
    (frozen_root / name).unlink()
    before = _files(frozen_root)
    with pytest.raises(ValueError, match="Day06 frozen reference rejected.*" + name):
        artifact.load_inference_reference(frozen_root)
    assert before == _files(frozen_root)


@pytest.mark.parametrize("name", [
    "configs/train_baseline.json", "data/processed/financial_phrasebank_75agree.csv",
    "reports/day04/validation_predictions.csv", "reports/day05/test_confusion_counts.png",
])
def test_checksum_mismatch_does_not_write(frozen_root: Path, name: str) -> None:
    path = frozen_root / name
    path.write_bytes(path.read_bytes() + b"\n")
    before = _files(frozen_root)
    with pytest.raises(ValueError, match="mismatch"):
        artifact.load_inference_reference(frozen_root)
    assert before == _files(frozen_root)


@pytest.mark.parametrize("field,value", [
    ("candidate_id", "linear_svc"), ("random_state", 99),
    ("preprocessing_policy", "different"), ("fit_policy", "train_plus_validation"),
    ("class_order", ["neutral", "negative", "positive"]),
    ("label_mapping", {"negative": 1, "neutral": 0, "positive": 2}),
    ("parameters", {}), ("pipeline_parameters", {}),
    ("runtime_versions", {"python": "different"}),
    ("validation_source_row_ids", []), ("train_source_row_ids", []),
])
def test_changed_frozen_contract_is_rejected(frozen_root: Path, field: str, value: object) -> None:
    payload = json.loads((frozen_root / "reports/day05/frozen_protocol.json").read_text())
    payload[field] = value
    _write_protocol(frozen_root, payload)
    before = _files(frozen_root)
    with pytest.raises(ValueError, match="Frozen protocol mismatch"):
        artifact.load_inference_reference(frozen_root)
    assert before == _files(frozen_root)


@pytest.mark.parametrize("damage", ["reorder", "missing", "duplicate", "truth"])
def test_validation_semantics_checked_even_with_updated_checksum(frozen_root: Path, damage: str) -> None:
    path = frozen_root / "reports/day04/validation_predictions.csv"
    frame = pd.read_csv(path)
    if damage == "reorder":
        frame = frame.iloc[::-1]
    elif damage == "missing":
        frame = frame.iloc[1:]
    elif damage == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    else:
        frame.loc[0, "true_label_id"] = (int(frame["true_label_id"].iloc[0]) + 1) % 3
    frame.to_csv(path, index=False)
    provenance_path = frozen_root / "reports/day04/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["output_checksums"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(provenance))
    before = _files(frozen_root)
    with pytest.raises(ValueError, match="unique|order|true labels"):
        artifact.load_inference_reference(frozen_root)
    assert before == _files(frozen_root)


def test_different_live_runtime_is_rejected(frozen_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    versions = final.runtime_versions()
    monkeypatch.setattr(final, "runtime_versions", lambda: {**versions, "joblib": "different"})
    with pytest.raises(ValueError, match="Frozen protocol mismatch"):
        artifact.load_inference_reference(frozen_root)


def test_current_preprocessing_policy_is_checked(frozen_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact.preprocess, "PREPROCESSING_POLICY", "different")
    with pytest.raises(ValueError, match="runtime preprocessing_policy"):
        artifact.load_inference_reference(frozen_root)


def test_input_change_between_readers_is_rejected(frozen_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = artifact.load_day04_inputs

    def change_after_read(root: Path):
        inputs = original(root)
        path = root / "reports/day05/final_evaluation.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return inputs

    monkeypatch.setattr(artifact, "load_day04_inputs", change_after_read)
    with pytest.raises(ValueError, match="Frozen input changed.*final_evaluation.json"):
        artifact.load_inference_reference(frozen_root)


def test_import_has_no_artifact_io(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", _forbid)
        context.setattr(Path, "read_text", _forbid)
        context.setattr(Pipeline, "fit", _forbid)
        context.setattr(Pipeline, "predict", _forbid)
        importlib.reload(artifact)
