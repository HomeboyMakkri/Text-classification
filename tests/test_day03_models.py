"""Estimator construction stays separate from fitting."""

from pathlib import Path
import sys

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finnews_sentiment.data.baseline_config import load_baseline_config
from finnews_sentiment.models.train_model import build_baseline_pipeline, build_dummy_classifier


def test_builders_follow_config_and_return_independent_unfitted_estimators() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_baseline_config(root / "configs/train_baseline.json")

    first = build_baseline_pipeline(config)
    second = build_baseline_pipeline(config)
    dummy_first = build_dummy_classifier(config)
    dummy_second = build_dummy_classifier(config)

    assert isinstance(first, Pipeline)
    assert list(first.named_steps) == ["tfidf", "clf"]
    vectorizer = first.named_steps["tfidf"]
    classifier = first.named_steps["clf"]
    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(classifier, LogisticRegression)
    vectorizer_params = vectorizer.get_params(deep=False)
    assert all(vectorizer_params[key] == value for key, value in config.tfidf.items())
    classifier_params = classifier.get_params(deep=False)
    assert all(classifier_params[key] == value for key, value in config.logreg.items())
    assert first is not second
    assert first.named_steps["tfidf"] is not second.named_steps["tfidf"]
    assert first.named_steps["clf"] is not second.named_steps["clf"]
    assert not hasattr(vectorizer, "vocabulary_")
    assert not hasattr(vectorizer, "idf_")
    assert not hasattr(classifier, "classes_")
    assert isinstance(dummy_first, DummyClassifier)
    assert dummy_first.get_params(deep=False)["strategy"] == config.dummy["strategy"]
    assert dummy_first is not dummy_second
    assert not hasattr(dummy_first, "classes_")
