"""Day04 candidate builders with explicit, reproducible estimator settings."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from typing import cast

from finnews_sentiment.data.baseline_config import BaselineConfig


def build_linear_svc_pipeline(config: BaselineConfig) -> Pipeline:
    """Return an unfitted baseline-TF-IDF and fixed-parameter LinearSVC Pipeline."""
    tfidf = TfidfVectorizer(**config.tfidf)
    classifier = LinearSVC(
        penalty="l2",
        loss="squared_hinge",
        # sklearn 1.9 accepts "auto"; the pinned stubs still declare bool only.
        dual=cast(bool, "auto"),
        tol=1e-4,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=42,
        max_iter=5000,
    )
    return Pipeline([("tfidf", tfidf), ("clf", classifier)])
