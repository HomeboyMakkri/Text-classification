"""Trusted bundle serialization, preflight, roundtrip and reuse boundaries."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.models import inference_artifact as artifact
from test_day06_reference import completed_reference, day04_reference


def _forbid(*args: object, **kwargs: object) -> None:
    raise AssertionError("This operation must not fit, predict, deserialize or rebuild")


def _files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def saved_pair(completed_reference: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    reference = artifact.load_inference_reference(completed_reference)
    bundle = artifact.build_inference_bundle(reference)
    return artifact.save_bundle(bundle, tmp_path_factory.mktemp("day06-artifact") / "bundle", reference=reference)


@pytest.fixture
def pair(saved_pair: Path, tmp_path: Path) -> Path:
    directory = tmp_path / "bundle"
    shutil.copytree(saved_pair.parent, directory)
    return directory / artifact.BUNDLE_NAME


def _manifest(path: Path) -> dict:
    return json.loads(path.with_name(artifact.MANIFEST_NAME).read_text())


def _write_manifest(path: Path, payload: dict) -> None:
    path.with_name(artifact.MANIFEST_NAME).write_text(json.dumps(payload))


def _rewrite_bundle(path: Path, bundle: object) -> None:
    joblib.dump(bundle, path, compress=3)
    manifest = _manifest(path)
    manifest["bundle_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_manifest(path, manifest)


def test_load_without_model_calls_or_training_artifacts(pair: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    monkeypatch.setattr(artifact, "load_inference_reference", _forbid)
    before = _files(pair.parent)
    bundle = artifact.load_bundle(pair)
    assert bundle.metadata.validation_replay == "exact_match"
    assert bundle.metadata.contract.candidate_id == "baseline"
    assert bundle.metadata.n_features == len(bundle.pipeline.named_steps["tfidf"].vocabulary_)
    assert _files(pair.parent) == before


@pytest.mark.parametrize("damage", ["schema", "hash", "runtime", "policy", "implementation", "classes", "dimensions", "ids", "metadata_schema"])
def test_manifest_preflight_precedes_unpickle(pair: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    manifest = _manifest(pair)
    contract = manifest["metadata"]["contract"]
    if damage == "schema":
        manifest["schema_version"] = "future"
    elif damage == "hash":
        pair.write_bytes(pair.read_bytes() + b"corrupt")
    elif damage == "runtime":
        contract["runtime_versions"]["scikit-learn"] = "different"
    elif damage == "policy":
        contract["fit_policy"] = "train_plus_validation"
    elif damage == "implementation":
        contract["preprocessing_sha256"] = "0" * 64
    elif damage == "classes":
        contract["label_mapping"]["negative"] = True
    elif damage == "dimensions":
        manifest["metadata"]["n_features"] = 0
    elif damage == "ids":
        contract["train_source_row_ids"] = contract["train_source_row_ids"][::-1]
    else:
        manifest["metadata"]["extra"] = "unexpected"
    _write_manifest(pair, manifest)
    monkeypatch.setattr(joblib, "load", _forbid)
    before = _files(pair.parent)
    with pytest.raises(ValueError, match="Cannot load Day06 artifact"):
        artifact.load_bundle(pair)
    assert _files(pair.parent) == before


@pytest.mark.parametrize("damage", ["unfitted", "wrong_type", "classes", "idf", "weights", "vocabulary", "params", "metadata", "state"])
def test_loaded_object_checked_even_when_file_hash_matches(pair: Path, damage: str) -> None:
    bundle = artifact.load_bundle(pair)
    if damage == "unfitted":
        bundle = replace(bundle, pipeline=Pipeline([("tfidf", bundle.pipeline.named_steps["tfidf"]),
                                                   ("clf", type(bundle.pipeline.named_steps["clf"])())]))
    elif damage == "wrong_type":
        _rewrite_bundle(pair, {"model": bundle.pipeline})
    elif damage == "classes":
        bundle.pipeline.named_steps["clf"].classes_ = np.array([2, 1, 0])
    elif damage == "idf":
        # Emulate damaged pickle state, bypassing the vectorizer's public setter.
        bundle.pipeline.named_steps["tfidf"]._tfidf.idf_ = np.ones(1)
    elif damage == "weights":
        bundle.pipeline.named_steps["clf"].coef_ = np.zeros((3, 1))
    elif damage == "vocabulary":
        bundle.pipeline.named_steps["tfidf"].vocabulary_["new"] = 99999
    elif damage == "params":
        bundle.pipeline.set_params(tfidf__lowercase=True)
    elif damage == "metadata":
        bundle = replace(bundle, metadata=replace(bundle.metadata, contract=replace(bundle.metadata.contract, random_state=99)))
    elif damage == "state":
        bundle.pipeline.named_steps["clf"].coef_[0, 0] += 0.125
    if damage != "wrong_type":
        _rewrite_bundle(pair, bundle)
    with pytest.raises(ValueError, match="Cannot load Day06 artifact"):
        artifact.load_bundle(pair)


def test_invalid_pickle_is_informative(pair: Path) -> None:
    pair.write_bytes(b"not a pickle")
    manifest = _manifest(pair)
    manifest["bundle_sha256"] = hashlib.sha256(pair.read_bytes()).hexdigest()
    _write_manifest(pair, manifest)
    with pytest.raises(ValueError, match="Cannot deserialize trusted bundle"):
        artifact.load_bundle(pair)


@pytest.mark.parametrize("missing", [artifact.BUNDLE_NAME, artifact.MANIFEST_NAME])
def test_partial_pair_never_rebuilds(completed_reference: Path, pair: Path, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    pair.with_name(missing).unlink()
    monkeypatch.setattr(artifact, "build_inference_bundle", _forbid)
    before = _files(pair.parent)
    with pytest.raises(ValueError, match="no automatic rebuild"):
        artifact.prepare_inference_artifact(completed_reference, pair.parent)
    assert _files(pair.parent) == before


def test_prepare_reuses_complete_pair_without_fit_predict_or_write(
    completed_reference: Path, pair: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _files(pair.parent)
    monkeypatch.setattr(artifact, "build_inference_bundle", _forbid)
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    monkeypatch.setattr(joblib, "dump", _forbid)
    assert artifact.prepare_inference_artifact(completed_reference, pair.parent) == pair
    assert _files(pair.parent) == before


def test_roundtrip_state_replay_and_manifest_last(
    completed_reference: Path, pair: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = artifact.load_inference_reference(completed_reference)
    bundle = artifact.load_bundle(pair)
    target = tmp_path / "new-artifact"
    original = Pipeline.predict
    calls = []

    def predict(self: Pipeline, X: list[str]) -> np.ndarray:
        assert X == reference.validation["text_clean"].tolist()
        assert not (target / artifact.MANIFEST_NAME).exists()
        calls.append("roundtrip_validation")
        return np.asarray(original(self, X))

    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "predict", predict)
    before = _files(completed_reference)
    path = artifact.save_bundle(bundle, target, reference=reference)
    loaded = artifact.load_bundle(path)
    assert calls == ["roundtrip_validation"]
    assert artifact._state_sha256(bundle) == artifact._state_sha256(loaded)
    assert artifact._metadata_payload(bundle.metadata) == _manifest(path)["metadata"]
    assert set(_files(target)) == {artifact.BUNDLE_NAME, artifact.MANIFEST_NAME}
    assert _files(completed_reference) == before


def test_replay_failure_does_not_publish_pair(
    completed_reference: Path, pair: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = artifact.load_inference_reference(completed_reference)
    bundle = artifact.load_bundle(pair)
    target = tmp_path / "bad-replay"
    monkeypatch.setattr(Pipeline, "predict", lambda self, X: np.full(len(X), 2))
    with pytest.raises(ValueError, match="Roundtrip validation replay mismatch"):
        artifact.save_bundle(bundle, target, reference=reference)
    assert not target.exists()
    assert not list(tmp_path.glob("day06-staging-*"))


def test_changed_input_blocks_reuse(completed_reference: Path, pair: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "project"
    shutil.copytree(completed_reference, root)
    (root / "configs/train_baseline.json").write_text("{}")
    monkeypatch.setattr(artifact, "build_inference_bundle", _forbid)
    before = _files(pair.parent)
    with pytest.raises(ValueError):
        artifact.prepare_inference_artifact(root, pair.parent)
    assert _files(pair.parent) == before


def test_save_rejects_unfitted_classifier_before_writing(completed_reference: Path, pair: Path, tmp_path: Path) -> None:
    reference = artifact.load_inference_reference(completed_reference)
    bundle = artifact.load_bundle(pair)
    bundle.pipeline.steps[-1] = ("clf", type(bundle.pipeline.named_steps["clf"])())
    output = tmp_path / "unfitted"
    with pytest.raises(ValueError, match="not fitted"):
        artifact.save_bundle(bundle, output, reference=reference)
    assert not output.exists()


def test_interrupted_publication_leaves_partial_pair_and_blocks_rebuild(
    completed_reference: Path, pair: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = artifact.load_inference_reference(completed_reference)
    bundle = artifact.load_bundle(pair)
    output = tmp_path / "interrupted"
    original = Path.open

    def fail_manifest(self: Path, *args, **kwargs):
        if self == output / artifact.MANIFEST_NAME and args == ("xb",):
            raise OSError("fixture interrupted manifest write")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_manifest)
    with pytest.raises(OSError, match="interrupted manifest write"):
        artifact.save_bundle(bundle, output, reference=reference)
    assert set(_files(output)) == {artifact.BUNDLE_NAME}
    monkeypatch.setattr(artifact, "build_inference_bundle", _forbid)
    before = _files(output)
    with pytest.raises(ValueError, match="no automatic rebuild"):
        artifact.prepare_inference_artifact(completed_reference, output)
    assert _files(output) == before


def test_cli_pickle_loads_in_separate_process(completed_reference: Path, tmp_path: Path) -> None:
    output = tmp_path / "cli-artifact"
    command = [str(ROOT / "venv/bin/python"), "-m", "finnews_sentiment.models.inference_artifact",
               "--root", str(completed_reference), "--output-dir", str(output)]
    import os
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    code = """
import sys
from pathlib import Path
from finnews_sentiment.models.inference_artifact import load_bundle
from unittest.mock import patch
def forbidden(*args, **kwargs):
    raise AssertionError('standalone load must not access training artifacts or model calls')
with patch('finnews_sentiment.models.inference_artifact.load_inference_reference', forbidden), \\
     patch('sklearn.pipeline.Pipeline.fit', forbidden), patch('sklearn.pipeline.Pipeline.predict', forbidden):
    bundle = load_bundle(Path(sys.argv[1]))
    assert type(bundle).__module__ == 'finnews_sentiment.models.inference_artifact'
assert 'matplotlib' not in sys.modules
print(bundle.metadata.validation_replay)
"""
    loaded = subprocess.run([str(ROOT / "venv/bin/python"), "-c", code, str(output / artifact.BUNDLE_NAME)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert loaded.returncode == 0, loaded.stderr
    assert "exact_match" in loaded.stdout
