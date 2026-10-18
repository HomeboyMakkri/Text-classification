"""Day04 classifier builders remain explicit and unfitted until called."""

from pathlib import Path
import sys
from typing import cast

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.data.baseline_config import load_baseline_config
from finnews_sentiment.models.experiment_models import build_linear_svc_pipeline


def test_linear_svc_builder_returns_independent_unfitted_baseline_pipelines() -> None:
    config = load_baseline_config(ROOT / "configs/train_baseline.json")

    first = build_linear_svc_pipeline(config)
    second = build_linear_svc_pipeline(config)

    assert isinstance(first, Pipeline) and isinstance(second, Pipeline)
    assert first is not second
    assert first.named_steps["tfidf"] is not second.named_steps["tfidf"]
    assert first.named_steps["clf"] is not second.named_steps["clf"]
    vectorizer = first.named_steps["tfidf"]
    classifier = first.named_steps["clf"]
    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(classifier, LinearSVC)
    vectorizer_params = vectorizer.get_params()
    assert vectorizer_params["ngram_range"] == config.tfidf["ngram_range"]
    assert vectorizer_params["max_features"] == config.tfidf["max_features"]
    assert vectorizer_params["analyzer"] == "word"
    assert classifier.get_params() == {
        "C": 1.0,
        "class_weight": None,
        "dual": "auto",
        "fit_intercept": True,
        "intercept_scaling": 1,
        "loss": "squared_hinge",
        "max_iter": 5000,
        "multi_class": "ovr",
        "penalty": "l2",
        "random_state": 42,
        "tol": 1e-4,
        "verbose": 0,
    }
    assert not hasattr(vectorizer, "vocabulary_")
    assert not hasattr(classifier, "classes_")


def test_linear_svc_pipeline_fits_three_classes_and_predicts_aligned_rows() -> None:
    config = load_baseline_config(ROOT / "configs/train_baseline.json")
    pipeline = build_linear_svc_pipeline(config)
    texts = [
        "loss outlook negative",
        "decline outlook negative",
        "steady outlook neutral",
        "unchanged outlook neutral",
        "growth outlook positive",
        "profit outlook positive",
    ]
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    pipeline.fit(texts, labels)
    # Single-label input guarantees ndarray; stubs also allow multi-output tuples.
    predictions = cast(np.ndarray, pipeline.predict(texts))

    assert np.array_equal(pipeline.named_steps["clf"].classes_, np.asarray([0, 1, 2]))
    assert predictions.shape == labels.shape
    assert set(predictions.tolist()) == {0, 1, 2}
    assert hasattr(pipeline.named_steps["tfidf"], "vocabulary_")
