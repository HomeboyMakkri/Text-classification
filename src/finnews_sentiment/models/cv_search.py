"""Bounded LogisticRegression C search on fixed Day04 train folds."""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from finnews_sentiment.data.experiment_splits import CVFoldSet
from finnews_sentiment.models.cv_runner import _validate_folds
from finnews_sentiment.models.evaluate import CLASS_IDS


LOGREG_C_VALUES = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class CSearchResult:
    """Fold scores and run context for the fixed three-value C search."""

    scores: pd.DataFrame
    parameters: dict[str, object]
    warnings: tuple[str, ...]
    n_splits: int
    random_state: int


def search_logreg_c(
    pipeline: Pipeline,
    train: pd.DataFrame,
    fold_set: CVFoldSet,
) -> CSearchResult:
    """Evaluate the fixed LogisticRegression C grid without refitting a winner."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("C search requires an sklearn Pipeline")
    try:
        check_is_fitted(pipeline)
    except NotFittedError:
        pass
    else:
        raise ValueError("C search Pipeline must be unfitted")
    classifier = pipeline.named_steps.get("clf")
    if not isinstance(classifier, LogisticRegression):
        raise ValueError("C search Pipeline must contain LogisticRegression as step 'clf'")
    _validate_folds(train, fold_set)

    fixed_folds = [
        (
            np.asarray(fold.train_positions, dtype=np.int64),
            np.asarray(fold.validation_positions, dtype=np.int64),
        )
        for fold in fold_set.folds
    ]
    scorers = {
        "macro_f1": make_scorer(
            f1_score,
            labels=list(CLASS_IDS),
            average="macro",
            zero_division=0,
        ),
        "accuracy": make_scorer(accuracy_score),
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"clf__C": list(LOGREG_C_VALUES)},
        scoring=scorers,
        cv=fixed_folds,
        refit=False,
        error_score="raise",
        return_train_score=False,
    )
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            search.fit(
                train["text_clean"].tolist(),
                train["label_id"].to_numpy(dtype=np.int64),
            )
        except ConvergenceWarning as error:
            raise RuntimeError("LogisticRegression C search raised ConvergenceWarning") from error
        except Exception as error:
            raise RuntimeError(f"LogisticRegression C search failed: {error}") from error
    captured_warnings.extend(
        f"{item.category.__name__}: {item.message}" for item in recorded_warnings
    )

    cv_results = search.cv_results_
    score_columns = {
        f"fold_{fold_id}_{metric}": np.asarray(
            cv_results[f"split{fold_id}_test_{metric}"], dtype=np.float64,
        )
        for metric in ("macro_f1", "accuracy")
        for fold_id in range(fold_set.n_splits)
    }
    scores = pd.DataFrame({"C": [float(params["clf__C"]) for params in cv_results["params"]]})
    for column, values in score_columns.items():
        scores[column] = values
    for metric in ("macro_f1", "accuracy"):
        fold_columns = [f"fold_{fold_id}_{metric}" for fold_id in range(fold_set.n_splits)]
        fold_matrix = scores[fold_columns].to_numpy(dtype=np.float64)
        scores[f"mean_{metric}"] = fold_matrix.mean(axis=1)
        scores[f"std_{metric}"] = fold_matrix.std(axis=1, ddof=0)
    scores["mean_fit_seconds"] = np.asarray(cv_results["mean_fit_time"], dtype=np.float64)
    return CSearchResult(
        scores=scores,
        parameters=pipeline.get_params(deep=True),
        warnings=tuple(captured_warnings),
        n_splits=fold_set.n_splits,
        random_state=fold_set.random_state,
    )
