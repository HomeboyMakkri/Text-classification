"""Estimator construction stays separate from fitting."""

from pathlib import Path
import sys

import pandas as pd
import pytest
import numpy as np
from scipy.sparse import issparse
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finnews_sentiment.data.baseline_config import BaselineConfig, load_baseline_config
from finnews_sentiment.data.split_data import NewsSplits, split_processed_news
from finnews_sentiment.models import train_model
from finnews_sentiment.models.train_model import build_baseline_pipeline, build_dummy_classifier, fit_baselines


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


def _training_fixture() -> tuple[pd.DataFrame, NewsSplits]:
    rows = [
        (i, f"company item{i} outlook {['loss', 'steady', 'growth'][i % 3]}",
         ("negative", "neutral", "positive")[i % 3],
         f"company item{i} outlook {['loss', 'steady', 'growth'][i % 3]}", i % 3)
        for i in range(90)
    ]
    news = pd.DataFrame(rows, columns=["source_row_id", "text", "sentiment", "text_clean", "label_id"])
    first_split = split_processed_news(news)
    validation_id = int(first_split.splits["validation"]["source_row_id"].iloc[0])
    mask = news["source_row_id"] == validation_id
    news.loc[mask, "text"] = news.loc[mask, "text"] + " validationonlymarker"
    news.loc[mask, "text_clean"] = news.loc[mask, "text_clean"] + " validationonlymarker"
    repeated_row = news.loc[[0]].assign(source_row_id=90)
    news = pd.concat([news, repeated_row], ignore_index=True)
    return news, split_processed_news(news)


def test_fit_uses_train_only_and_returns_aligned_validation_predictions() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_baseline_config(root / "configs/train_baseline.json")
    _, splits = _training_fixture()

    result = fit_baselines(splits, config)
    train = splits.splits["train"]
    validation = splits.splits["validation"]
    vectorizer = result.pipeline.named_steps["tfidf"]
    idf_before_predict = vectorizer.idf_.copy()
    vocabulary = vectorizer.vocabulary_

    assert result.train_feature_shape[0] == len(train)
    assert result.feature_count == result.train_feature_shape[1] <= 5000
    assert issparse(vectorizer.transform(train["text_clean"].tolist()))
    assert "validationonlymarker" not in vocabulary
    assert result.pipeline.named_steps["clf"].classes_.tolist() == [0, 1, 2]
    assert tuple(result.dummy.classes_) == (0, 1, 2)
    assert result.convergence_status == "converged_no_warning"
    predictions = result.validation_predictions
    assert predictions["source_row_id"].tolist() == validation["source_row_id"].tolist()
    assert len(predictions) == len(validation)
    for column in ("baseline_prediction", "dummy_prediction"):
        assert set(predictions[column]).issubset({0, 1, 2})
    repeated_predictions = result.pipeline.predict(validation["text_clean"].tolist())
    np.testing.assert_array_equal(repeated_predictions, predictions["baseline_prediction"].to_numpy())
    assert vectorizer.vocabulary_ == vocabulary
    np.testing.assert_array_equal(vectorizer.idf_, idf_before_predict)

    repeated = fit_baselines(splits, config)
    pd.testing.assert_frame_equal(predictions, repeated.validation_predictions)
    assert repeated.train_feature_shape == result.train_feature_shape


def test_malformed_manifest_fails_before_building_estimators(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_baseline_config(root / "configs/train_baseline.json")
    _, splits = _training_fixture()
    bad_manifest = pd.concat([splits.manifest, splits.manifest.iloc[[0]]], ignore_index=True)
    malformed = NewsSplits(bad_manifest, splits.splits, splits.summary)

    def should_not_build(_: BaselineConfig) -> Pipeline:
        raise AssertionError("Estimator construction must happen after split validation")

    monkeypatch.setattr(train_model, "build_baseline_pipeline", should_not_build)
    with pytest.raises(ValueError, match="manifest source_row_id values must be unique"):
        fit_baselines(malformed, config)
