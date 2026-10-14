"""Typed baseline configuration; no estimator construction, split or fit."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Literal, TypedDict


class TfidfSettings(TypedDict):
    max_features: int
    ngram_range: tuple[int, int]
    analyzer: Literal["word"]
    lowercase: bool
    stop_words: None
    min_df: int
    max_df: float
    use_idf: bool
    smooth_idf: bool
    sublinear_tf: bool
    norm: Literal["l2"]
    token_pattern: str
    strip_accents: None


class LogRegSettings(TypedDict):
    solver: Literal["lbfgs"]
    C: float
    l1_ratio: float
    max_iter: int
    tol: float
    class_weight: None
    random_state: int


class DummySettings(TypedDict):
    strategy: Literal["most_frequent"]


@dataclass(frozen=True)
class BaselineConfig:
    policy: str
    input_path: Path
    provenance_path: Path
    text_col: str
    target_col: str
    random_state: int
    test_size: float
    validation_size: float
    tfidf: TfidfSettings
    logreg: LogRegSettings
    dummy: DummySettings
    split_output_dir: Path
    report_output_dir: Path


def _object(value: object, name: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must be an object with exactly these keys: {sorted(keys)}")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        result[key] = item
    return result


def _fixed(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} must be {expected!r} for baseline-v1")


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, upper: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0 or (upper is not None and result > upper):
        raise ValueError(f"{name} must be positive and finite, upper bound={upper}")
    return result


def _fraction(value: object, name: str) -> float:
    result = _number(value, name, upper=1.0)
    if result == 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return result


def _path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty project-relative path")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{name} must stay relative to the project root, without '..'")
    return path


def load_baseline_config(path: Path) -> BaselineConfig:
    """Validate settings without importing sklearn or touching data.

    Numeric settings are explicit and validated for future typed builders.
    Day03 freezes the checked-in values before measuring validation metrics.
    """
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    config = _object(payload, "baseline config", {
        "policy", "input_path", "provenance_path", "text_col", "target_col",
        "random_state", "test_size", "validation_size", "tfidf", "logreg", "dummy",
        "split_output_dir", "report_output_dir",
    })
    for key, expected in (("policy", "baseline-v1"), ("text_col", "text_clean"), ("target_col", "label_id")):
        _fixed(config[key], expected, key)
    seed = _integer(config["random_state"], "random_state", minimum=0)
    if seed > 2**32 - 1:
        raise ValueError("random_state must be <= 2**32 - 1")

    tfidf = _object(config["tfidf"], "tfidf", {
        "max_features", "ngram_range", "analyzer", "lowercase", "stop_words", "min_df", "max_df",
        "use_idf", "smooth_idf", "sublinear_tf", "norm", "token_pattern", "strip_accents",
    })
    token_pattern = r"(?u)\b\w\w+\b"
    for key, expected in (
        ("analyzer", "word"), ("lowercase", False), ("stop_words", None),
        ("use_idf", True), ("smooth_idf", True), ("sublinear_tf", False),
        ("norm", "l2"), ("token_pattern", token_pattern), ("strip_accents", None),
    ):
        _fixed(tfidf[key], expected, f"tfidf.{key}")
    ngrams = tfidf["ngram_range"]
    if not isinstance(ngrams, list) or len(ngrams) != 2:
        raise ValueError("tfidf.ngram_range must contain two positive integers")
    low = _integer(ngrams[0], "tfidf.ngram_range[0]")
    high = _integer(ngrams[1], "tfidf.ngram_range[1]")
    if low > high:
        raise ValueError("tfidf.ngram_range lower bound exceeds upper bound")
    tfidf_settings: TfidfSettings = {
        "max_features": _integer(tfidf["max_features"], "tfidf.max_features"),
        "ngram_range": (low, high), "analyzer": "word", "lowercase": False,
        "stop_words": None, "min_df": _integer(tfidf["min_df"], "tfidf.min_df"),
        "max_df": _number(tfidf["max_df"], "tfidf.max_df", upper=1.0),
        "use_idf": True, "smooth_idf": True, "sublinear_tf": False,
        "norm": "l2", "token_pattern": token_pattern, "strip_accents": None,
    }
    logreg = _object(config["logreg"], "logreg", {
        "solver", "C", "l1_ratio", "max_iter", "tol", "class_weight", "random_state",
    })
    for key, expected in (("solver", "lbfgs"), ("l1_ratio", 0.0), ("class_weight", None), ("random_state", seed)):
        _fixed(logreg[key], expected, f"logreg.{key}")
    logreg_settings: LogRegSettings = {
        "solver": "lbfgs", "C": _number(logreg["C"], "logreg.C"), "l1_ratio": 0.0,
        "max_iter": _integer(logreg["max_iter"], "logreg.max_iter"),
        "tol": _number(logreg["tol"], "logreg.tol"), "class_weight": None, "random_state": seed,
    }
    dummy = _object(config["dummy"], "dummy", {"strategy"})
    _fixed(dummy["strategy"], "most_frequent", "dummy.strategy")
    return BaselineConfig(
        policy="baseline-v1", input_path=_path(config["input_path"], "input_path"),
        provenance_path=_path(config["provenance_path"], "provenance_path"),
        text_col="text_clean", target_col="label_id", random_state=seed,
        test_size=_fraction(config["test_size"], "test_size"),
        validation_size=_fraction(config["validation_size"], "validation_size"),
        tfidf=tfidf_settings, logreg=logreg_settings, dummy={"strategy": "most_frequent"},
        split_output_dir=_path(config["split_output_dir"], "split_output_dir"),
        report_output_dir=_path(config["report_output_dir"], "report_output_dir"),
    )
