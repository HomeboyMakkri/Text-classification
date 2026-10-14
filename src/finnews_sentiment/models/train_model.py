"""Builders for the unfitted Day03 baseline estimators."""

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from finnews_sentiment.data.baseline_config import BaselineConfig


def build_baseline_pipeline(config: BaselineConfig) -> Pipeline:
    """Return a new, unfitted TF-IDF and LogisticRegression pipeline."""
    tfidf = TfidfVectorizer(**config.tfidf)
    classifier = LogisticRegression(**config.logreg)
    return Pipeline([("tfidf", tfidf), ("clf", classifier)])


def build_dummy_classifier(config: BaselineConfig) -> DummyClassifier:
    """Return a new, unfitted majority-class DummyClassifier."""
    return DummyClassifier(**config.dummy)
