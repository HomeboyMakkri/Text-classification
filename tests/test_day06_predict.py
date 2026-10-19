"""Raw-text batch validation, preprocessing parity and immutable fitted state."""

from collections.abc import Sequence
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.features.preprocess import preprocess_text
from finnews_sentiment.models import inference_artifact as artifact
from finnews_sentiment.models import predict
from sklearn.pipeline import Pipeline
from test_day06_artifact import saved_pair, completed_reference, day04_reference


def _forbid(*args: object, **kwargs: object) -> None:
    raise AssertionError("Inference must not fit, use confidence APIs, access training references or network")


@pytest.fixture
def bundle(saved_pair: Path, monkeypatch: pytest.MonkeyPatch) -> artifact.InferenceBundle:
    result = artifact.load_bundle(saved_pair)
    for target in (Pipeline, type(result.pipeline.named_steps["tfidf"]), type(result.pipeline.named_steps["clf"])):
        for name in ("fit", "fit_transform", "partial_fit"):
            monkeypatch.setattr(target, name, _forbid, raising=False)
    monkeypatch.setattr(Pipeline, "predict_proba", _forbid)
    monkeypatch.setattr(Pipeline, "decision_function", _forbid)
    monkeypatch.setattr(artifact, "load_inference_reference", _forbid)
    monkeypatch.setattr("socket.create_connection", _forbid)
    monkeypatch.setattr("urllib.request.urlopen", _forbid)
    return result


def test_full_batch_cleaned_once_order_duplicates_and_state(
    bundle: artifact.InferenceBundle, monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = ["  Profit is NOT down -5% &amp; costs £2.50.  ", "<b>No</b> growth in 2026.",
             "  Profit is NOT down -5% &amp; costs £2.50.  "]
    original_texts = texts.copy()
    state = artifact._state_sha256(bundle)
    before = asdict(bundle.metadata)
    original = Pipeline.predict
    calls: list[list[str]] = []

    def spy(self: Pipeline, X: list[str]) -> np.ndarray:
        calls.append(X.copy())
        return np.asarray(original(self, X))

    monkeypatch.setattr(Pipeline, "predict", spy)
    results = predict.predict_texts(bundle, texts)
    assert calls == [[preprocess_text(text) for text in texts]]
    assert results[0].text_clean == "profit is not down -5% & costs £2.50."
    assert results[1].text_clean == "no growth in 2026."
    assert [item.text for item in results] == texts
    assert results[0] == results[2]
    assert len(results) == len(texts) and texts == original_texts
    assert all(type(item.pred_label_id) is int for item in results)
    assert all(item.pred_label == ("negative", "neutral", "positive")[item.pred_label_id] for item in results)
    assert all(set(asdict(item)) == {"text", "text_clean", "pred_label_id", "pred_label"} for item in results)
    assert artifact._state_sha256(bundle) == state
    assert asdict(bundle.metadata) == before
    assert predict.predict_texts(bundle, tuple(texts)) == results
    assert predict.predict_texts(bundle, [texts[0]]) == [results[0]]
    assert artifact._state_sha256(bundle) == state


@pytest.mark.parametrize("texts,error,pattern", [
    ("one text", TypeError, "Sequence"), (b"one text", TypeError, "Sequence"),
    (None, TypeError, "Sequence"), ({"text": "value"}, TypeError, "Sequence"),
    ({"a", "b"}, TypeError, "Sequence"), (iter(["text"]), TypeError, "Sequence"),
    ([], ValueError, "empty"), ((), ValueError, "empty"),
    (["valid", None], TypeError, r"texts\[1\]"),
    (["valid", float("nan")], TypeError, r"texts\[1\]"),
    (["valid", 42], TypeError, r"texts\[1\]"),
    (["valid", b"text"], TypeError, r"texts\[1\]"),
    (["valid", ""], ValueError, r"texts\[1\]"),
    (["valid", " \t\n"], ValueError, r"texts\[1\]"),
    (["valid", "<b></b>"], ValueError, r"texts\[1\]"),
    (["valid", "&nbsp;"], ValueError, r"texts\[1\]"),
])
def test_rejects_whole_batch_before_model_call(
    bundle: artifact.InferenceBundle, monkeypatch: pytest.MonkeyPatch,
    texts: Sequence[str], error: type[Exception], pattern: str,
) -> None:
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    state = artifact._state_sha256(bundle)
    with pytest.raises(error, match=pattern):
        predict.predict_texts(bundle, texts)
    assert artifact._state_sha256(bundle) == state


@pytest.mark.parametrize("label,name", [(0, "negative"), (1, "neutral"), (2, "positive")])
def test_fixed_label_mapping(bundle: artifact.InferenceBundle, monkeypatch: pytest.MonkeyPatch, label: int, name: str) -> None:
    monkeypatch.setattr(Pipeline, "predict", lambda self, X: np.array([label], dtype="int64"))
    item = predict.predict_texts(bundle, ["An unlabeled financial sentence."])[0]
    assert item.pred_label_id == label and item.pred_label == name


@pytest.mark.parametrize("output", [np.array([3]), np.array([1.0]), np.array([True]), np.array([[1]]),
                                    np.array([], dtype="int64"), np.array([0, 1])])
def test_invalid_model_output_is_not_returned(
    bundle: artifact.InferenceBundle, monkeypatch: pytest.MonkeyPatch, output: np.ndarray,
) -> None:
    monkeypatch.setattr(Pipeline, "predict", lambda self, X: output)
    with pytest.raises(ValueError, match="class ID|one-dimensional|length mismatch"):
        predict.predict_texts(bundle, ["Unlabeled text"])


def test_oov_zero_row_is_valid_without_refit(bundle: artifact.InferenceBundle) -> None:
    text = "zxqvplmnbvcxqz"
    vectorizer = bundle.pipeline.named_steps["tfidf"]
    assert text not in vectorizer.vocabulary_
    matrix = vectorizer.transform([text])
    assert matrix.shape == (1, bundle.metadata.n_features) and matrix.nnz == 0
    state = artifact._state_sha256(bundle)
    result = predict.predict_texts(bundle, [text])[0]
    assert result.text == text and result.text_clean == text
    assert result.pred_label_id in (0, 1, 2)
    assert artifact._state_sha256(bundle) == state


def test_string_nan_na_and_numbers_are_preserved(bundle: artifact.InferenceBundle) -> None:
    results = predict.predict_texts(bundle, ["nan", "NA", "123.45 €"])
    assert [item.text for item in results] == ["nan", "NA", "123.45 €"]
    assert [item.text_clean for item in results] == ["nan", "na", "123.45 €"]


def test_mutated_class_order_is_rejected_before_predict(bundle: artifact.InferenceBundle, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle.pipeline.named_steps["clf"].classes_ = np.array([2, 1, 0])
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    with pytest.raises(ValueError, match="Fitted classes"):
        predict.predict_texts(bundle, ["New text"])


def test_csv_boundary_does_not_read_or_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "read_bytes", _forbid)
    with pytest.raises(NotImplementedError, match="D6-06"):
        predict.predict_for_file(tmp_path / "model", tmp_path / "input.csv", tmp_path / "output.csv")
    assert not list(tmp_path.iterdir())


def test_csv_cli_reports_pending_stage(tmp_path: Path) -> None:
    command = [str(ROOT / "venv/bin/python"), "-m", "finnews_sentiment.models.predict", "--model", "missing.joblib",
               "--input", "missing.csv", "--output", "output.csv"]
    result = subprocess.run(command, cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                            capture_output=True, text=True)
    assert result.returncode == 2 and "D6-06" in result.stderr
    assert "Traceback" not in result.stderr and not (tmp_path / "output.csv").exists()
