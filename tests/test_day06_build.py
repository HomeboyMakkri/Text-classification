"""Explicit Day06 fit boundary, frozen-parameter checks and fresh replay."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.models import inference_artifact as artifact
from finnews_sentiment.models import run_experiments_day04 as experiments
from test_day06_reference import completed_reference, day04_reference


@pytest.fixture
def reference(completed_reference: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> artifact.InferenceReference:
    root = tmp_path / "project"
    shutil.copytree(completed_reference, root)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Day06 build must not run CV/final evaluation/network")

    monkeypatch.setattr("finnews_sentiment.models.final_evaluation.run_final_evaluation", forbidden)
    monkeypatch.setattr("finnews_sentiment.models.cv_runner.run_cv_candidate", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    return artifact.load_inference_reference(root)


def _files(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def _forbid(*args: object, **kwargs: object) -> None:
    raise AssertionError("Invalid input must stop before fit/predict")


def test_train_only_fit_and_single_validation_replay(
    reference: artifact.InferenceReference, monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _files(reference.root)
    train_before = reference.train.copy(deep=True)
    valid_before = reference.validation.copy(deep=True)
    original_fit = Pipeline.fit
    original_predict = Pipeline.predict
    original_fit_transform = TfidfVectorizer.fit_transform
    calls: list[str] = []

    def fit(self: Pipeline, X: list[str], y: np.ndarray) -> Pipeline:
        assert calls == []
        assert X == reference.train["text_clean"].tolist()
        np.testing.assert_array_equal(y, reference.train["label_id"].to_numpy())
        calls.append("fit_train")
        return original_fit(self, X, y)

    def fit_transform(self: TfidfVectorizer, raw_documents: list[str], y: object = None):
        assert raw_documents == reference.train["text_clean"].tolist()
        transformed = original_fit_transform(self, raw_documents)
        assert issparse(transformed)
        assert transformed.shape[0] == len(reference.train)
        calls.append("tfidf_train_only")
        return transformed

    def predict(self: Pipeline, X: list[str]) -> np.ndarray:
        assert calls == ["fit_train", "tfidf_train_only"]
        assert X == reference.validation["text_clean"].tolist()
        calls.append("validation_replay")
        return np.asarray(original_predict(self, X))

    monkeypatch.setattr(Pipeline, "fit", fit)
    monkeypatch.setattr(Pipeline, "predict", predict)
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", fit_transform)
    bundle = artifact.build_inference_bundle(reference)
    assert calls == ["fit_train", "tfidf_train_only", "validation_replay"]
    assert bundle.metadata.n_train == len(reference.train)
    assert bundle.metadata.n_validation == len(reference.validation)
    assert bundle.metadata.n_features == reference.day05_n_features
    assert bundle.metadata.validation_replay == "exact_match"
    assert bundle.metadata.contract == reference.contract
    assert bundle.metadata.contract is not reference.contract
    assert bundle.metadata.validation_reference_sha256 == reference.contract.input_checksums[
        "reports/day04/validation_predictions.csv"
    ]
    vectorizer = bundle.pipeline.named_steps["tfidf"]
    for text in reference.validation["text_clean"]:
        token = next(word for word in text.split() if word.startswith("item"))
        assert token not in vectorizer.vocabulary_
    protocol = json.loads((reference.root / "reports/day05/frozen_protocol.json").read_text())
    processed = pd.read_csv(reference.root / "data/processed/financial_phrasebank_75agree.csv")
    test_rows = processed.loc[processed["source_row_id"].isin(protocol["test_source_row_ids"])]
    for text in test_rows["text_clean"]:
        token = next(word for word in text.split() if word.startswith("item"))
        assert token not in vectorizer.vocabulary_
    np.testing.assert_array_equal(bundle.pipeline.classes_, [0, 1, 2])
    pd.testing.assert_frame_equal(reference.train, train_before)
    pd.testing.assert_frame_equal(reference.validation, valid_before)
    assert _files(reference.root) == before
    assert not (reference.root / "models").exists()


@pytest.mark.parametrize("damage", ["train_order", "train_label", "raw_text", "clean_text",
                                    "validation_order", "prediction_order", "prediction_label", "truth"])
def test_changed_snapshot_rejected_before_fit(
    reference: artifact.InferenceReference, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    if damage == "train_order":
        reference = replace(reference, train=reference.train.iloc[::-1].copy())
    elif damage == "validation_order":
        reference = replace(reference, validation=reference.validation.iloc[::-1].copy())
    elif damage == "prediction_order":
        reference = replace(reference, validation_predictions=reference.validation_predictions.iloc[::-1].copy())
    else:
        frame = reference.train if damage in ("train_label", "raw_text", "clean_text") else reference.validation_predictions
        column = {"train_label": "label_id", "raw_text": "text", "clean_text": "text_clean",
                  "prediction_label": "pred_label_id", "truth": "true_label_id"}[damage]
        frame.loc[frame.index[0], column] = "changed" if "text" in column else 99
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    with pytest.raises(ValueError, match="reference .* differs from frozen artifacts"):
        artifact.build_inference_bundle(reference)


@pytest.mark.parametrize("damage", ["candidate", "seed", "parameters", "fit_policy", "feature_count"])
def test_changed_metadata_rejected_before_fit(
    reference: artifact.InferenceReference, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    if damage == "candidate":
        reference = replace(reference, contract=replace(reference.contract, candidate_id="linear_svc"))
    elif damage == "seed":
        reference = replace(reference, contract=replace(reference.contract, random_state=99))
    elif damage == "parameters":
        reference.contract.pipeline_parameters["tfidf"]["max_features"] = 1
    elif damage == "fit_policy":
        reference = replace(reference, contract=replace(reference.contract, fit_policy="train_plus_validation"))
    else:
        reference = replace(reference, day05_n_features=999)
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    with pytest.raises(ValueError, match="metadata differs"):
        artifact.build_inference_bundle(reference)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_raw_clean_parity_checked_before_fit(
    reference: artifact.InferenceReference, monkeypatch: pytest.MonkeyPatch, split: str,
) -> None:
    original = artifact.preprocess.preprocess_text
    target = str(getattr(reference, split)["text"].iloc[0])
    monkeypatch.setattr(artifact.preprocess, "preprocess_text",
                        lambda text: "changed cleanup" if text == target else original(text))
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    with pytest.raises(ValueError, match=f"Raw/clean preprocessing mismatch: {split}, source_row_id="):
        artifact.build_inference_bundle(reference)


def test_builder_parameters_rechecked_before_fit(
    reference: artifact.InferenceReference, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = experiments._candidate_pipeline

    def wrong_builder(candidate_id: str, config):
        assert candidate_id == reference.contract.candidate_id
        pipeline, selected = original(candidate_id, config)
        pipeline.set_params(tfidf__max_df=0.5)
        return pipeline, selected

    monkeypatch.setattr(experiments, "_candidate_pipeline", wrong_builder)
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    with pytest.raises(ValueError, match="pipeline_parameters"):
        artifact.build_inference_bundle(reference)


@pytest.mark.parametrize("failure", ["prediction", "unknown_label", "prediction_shape", "fit_warning", "replay_warning"])
def test_replay_and_convergence_failures_return_no_bundle(
    reference: artifact.InferenceReference, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    before = _files(reference.root)
    original_predict = Pipeline.predict
    if failure == "fit_warning":
        def warning_fit(*args: object, **kwargs: object) -> None:
            warnings.warn("fixture convergence", ConvergenceWarning)
        monkeypatch.setattr(Pipeline, "fit", warning_fit)
        monkeypatch.setattr(Pipeline, "predict", _forbid)
    else:
        def failed_predict(self: Pipeline, X: list[str]) -> np.ndarray:
            assert X == reference.validation["text_clean"].tolist()
            if failure == "replay_warning":
                warnings.warn("fixture convergence", ConvergenceWarning)
            predicted = np.asarray(original_predict(self, X))
            if failure == "prediction":
                return (predicted + 1) % 3
            if failure == "unknown_label":
                return predicted + 3
            return predicted.reshape(-1, 1)
        monkeypatch.setattr(Pipeline, "predict", failed_predict)
    with pytest.raises(ValueError, match="replay mismatch|unknown class ID|one-dimensional|fixture convergence"):
        artifact.build_inference_bundle(reference)
    assert before == _files(reference.root)


def test_replay_cannot_change_fitted_vocabulary(
    reference: artifact.InferenceReference, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Pipeline.predict

    def mutate(self: Pipeline, X: list[str]) -> np.ndarray:
        predicted = np.asarray(original(self, X))
        self.named_steps["tfidf"].vocabulary_["leaked_validation_token"] = 99999
        return predicted

    monkeypatch.setattr(Pipeline, "predict", mutate)
    with pytest.raises(ValueError, match="changed fitted vocabulary/IDF/classes"):
        artifact.build_inference_bundle(reference)


def test_wrong_fitted_class_order_stops_before_predict(
    reference: artifact.InferenceReference, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Pipeline.fit

    def wrong_classes(self: Pipeline, X: list[str], y: np.ndarray) -> Pipeline:
        result = original(self, X, y)
        self.named_steps["clf"].classes_ = np.array([1, 0, 2])
        return result

    monkeypatch.setattr(Pipeline, "fit", wrong_classes)
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    with pytest.raises(ValueError, match="Fitted classifier classes must be"):
        artifact.build_inference_bundle(reference)


def test_input_change_during_fit_is_detected(
    reference: artifact.InferenceReference, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Pipeline.fit

    def mutate(self: Pipeline, X: list[str], y: np.ndarray) -> Pipeline:
        result = original(self, X, y)
        path = reference.root / "configs/train_baseline.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(Pipeline, "fit", mutate)
    with pytest.raises(ValueError, match="Frozen input changed during Day06 build"):
        artifact.build_inference_bundle(reference)


def test_repeated_build_reproduces_state_and_predictions(reference: artifact.InferenceReference) -> None:
    before = _files(reference.root)
    first = artifact.build_inference_bundle(reference)
    second = artifact.build_inference_bundle(reference)
    assert first.metadata == second.metadata
    assert first.pipeline.named_steps["tfidf"].vocabulary_ == second.pipeline.named_steps["tfidf"].vocabulary_
    np.testing.assert_array_equal(first.pipeline.named_steps["tfidf"].idf_, second.pipeline.named_steps["tfidf"].idf_)
    np.testing.assert_array_equal(first.pipeline.named_steps["clf"].coef_, second.pipeline.named_steps["clf"].coef_)
    np.testing.assert_array_equal(first.pipeline.named_steps["clf"].intercept_, second.pipeline.named_steps["clf"].intercept_)
    texts = reference.validation["text_clean"].tolist()
    np.testing.assert_array_equal(first.pipeline.predict(texts), second.pipeline.predict(texts))
    assert before == _files(reference.root)
