"""Offline Day03 input boundaries; fixtures are never notebook data."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.data.baseline_config import load_baseline_config
from finnews_sentiment.data.load_processed import (
    PROCESSED_COLUMNS, load_processed_news, validate_processed_news,
)
from finnews_sentiment.features.labels_duplicates import DuplicateLabelConflictError


@dataclass(frozen=True)
class InputBundle:
    csv: Path
    provenance: Path
    dataset_config: Path
    frame: pd.DataFrame


@pytest.fixture
def bundle(tmp_path: Path) -> InputBundle:
    frame = pd.DataFrame({
        "source_row_id": [0, 1, 3, 4], "text": ["NA", "nan", "123", "NA"],
        "sentiment": ["neutral", "negative", "positive", "neutral"],
        "text_clean": ["na", "nan", "123", "na"], "label_id": [1, 0, 2, 1],
    })
    source = {
        "dataset": "fixture", "revision": "fixture-revision", "url": "https://example.invalid/source",
        "archive_sha256": "0" * 64, "subset": "fixture", "member": "Sentences.txt",
        "encoding": "iso-8859-1", "license": "fixture-license", "citation": "fixture-author",
        "expected_rows": 5,
    }
    config_path = tmp_path / "configs/dataset.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({**source, "class_order": ["negative", "neutral", "positive"]}))
    csv_path = tmp_path / "data/processed/financial_phrasebank_75agree.csv"
    csv_path.parent.mkdir(parents=True)
    frame.to_csv(csv_path, index=False)
    provenance = {
        "source": source, "dataset_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "input": {"path": "data/interim/financial_phrasebank_75agree.csv", "sha256": "1" * 64},
        "archive": {"path": "data/raw/FinancialPhraseBank-v1.0.zip", "sha256": "0" * 64},
        "preprocessing_policy": "minimal-v1",
        "parameters": {"duplicate_policy": "retain_same_label_block_conflicts", "source_row_id": "zero_based_canonical_position"},
        "class_order": ["negative", "neutral", "positive"], "label_mapping": {"negative": 0, "neutral": 1, "positive": 2},
        "versions": {"python": "fixture-python", "pandas": "fixture-pandas", "numpy": "fixture-numpy"},
        "output": {"path": str(csv_path.relative_to(tmp_path)), "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                   "rows": len(frame), "columns": list(PROCESSED_COLUMNS)},
    }
    provenance_path = tmp_path / "reports/day02/provenance.json"
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_text(json.dumps(provenance))
    return InputBundle(csv_path, provenance_path, config_path, frame)


def refresh_checksum(bundle: InputBundle) -> None:
    """Let malformed CSV tests reach schema/value checks after hash verification."""
    payload = json.loads(bundle.provenance.read_text())
    payload["output"]["sha256"] = hashlib.sha256(bundle.csv.read_bytes()).hexdigest()
    bundle.provenance.write_text(json.dumps(payload))


def test_literal_texts_gaps_duplicates_and_input_immutability(bundle: InputBundle) -> None:
    paths = (bundle.csv, bundle.provenance, bundle.dataset_config)
    before = [path.read_bytes() for path in paths]
    loaded = load_processed_news(*paths)
    pd.testing.assert_frame_equal(loaded, bundle.frame)
    assert loaded["text"].tolist() == ["NA", "nan", "123", "NA"]
    assert loaded["source_row_id"].tolist() == [0, 1, 3, 4]
    assert str(loaded["source_row_id"].dtype) == str(loaded["label_id"].dtype) == "int64"
    pd.testing.assert_frame_equal(load_processed_news(*paths), loaded)
    snapshot = loaded.copy(deep=True)
    loaded.index = pd.Index([7, 7, 2, 2])
    snapshot.index = loaded.index.copy()
    validate_processed_news(loaded)
    pd.testing.assert_frame_equal(loaded, snapshot)
    assert before == [path.read_bytes() for path in paths]


@pytest.mark.parametrize("missing", ["csv", "provenance"])
def test_missing_artifact_has_separate_preparation_command(bundle: InputBundle, missing: str) -> None:
    path = bundle.csv if missing == "csv" else bundle.provenance
    path.unlink()
    with pytest.raises(FileNotFoundError, match="finnews_sentiment.data.prepare_day02"):
        load_processed_news(bundle.csv, bundle.provenance, bundle.dataset_config)
    assert not path.exists()


@pytest.mark.parametrize("problem", [
    "policy", "mapping", "boolean_mapping", "class_order", "source", "dataset_hash", "csv_hash",
    "columns", "rows", "boolean_rows", "output_path", "parameters", "versions", "archive_hash", "canonical_hash",
])
def test_provenance_mismatches_fail(bundle: InputBundle, problem: str) -> None:
    data = json.loads(bundle.provenance.read_text())
    if problem == "policy":
        data["preprocessing_policy"] = "unapproved-v2"
    elif problem == "mapping":
        data["label_mapping"]["negative"] = 2
    elif problem == "boolean_mapping":
        data["label_mapping"]["negative"] = False
    elif problem == "class_order":
        data["class_order"].reverse()
    elif problem == "source":
        data["source"]["revision"] = "other"
    elif problem == "dataset_hash":
        data["dataset_config_sha256"] = "2" * 64
    elif problem == "csv_hash":
        data["output"]["sha256"] = "2" * 64
    elif problem == "columns":
        data["output"]["columns"].reverse()
    elif problem == "rows":
        data["output"]["rows"] = 3
    elif problem == "boolean_rows":
        data["output"]["rows"] = True
    elif problem == "output_path":
        data["output"]["path"] = "../other.csv"
    elif problem == "parameters":
        data["parameters"]["duplicate_policy"] = "drop"
    elif problem == "versions":
        del data["versions"]["pandas"]
    elif problem == "archive_hash":
        data["archive"]["sha256"] = "2" * 64
    else:
        data["input"]["sha256"] = "not-a-checksum"
    bundle.provenance.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_processed_news(bundle.csv, bundle.provenance, bundle.dataset_config)


@pytest.mark.parametrize("problem", ["missing", "extra", "reordered", "duplicate", "empty", "extra_field", "blank_record"])
def test_csv_schema_and_records_fail(bundle: InputBundle, problem: str) -> None:
    frame = bundle.frame.copy(deep=True)
    if problem == "missing":
        frame = frame.drop(columns="label_id")
    elif problem == "extra":
        frame["unexpected"] = 1
    elif problem == "reordered":
        frame = frame[list(reversed(PROCESSED_COLUMNS))]
    elif problem == "duplicate":
        frame.columns = pd.Index(["source_row_id", "text", "sentiment", "text_clean", "text"])
    elif problem == "empty":
        frame = frame.iloc[:0]
    frame.to_csv(bundle.csv, index=False)
    if problem == "extra_field":
        bundle.csv.write_text(bundle.csv.read_text().replace("0,NA,neutral,na,1", "0,NA,neutral,na,1,ignored"))
    elif problem == "blank_record":
        bundle.csv.write_text(bundle.csv.read_text() + "\n")
    refresh_checksum(bundle)
    with pytest.raises(ValueError):
        load_processed_news(bundle.csv, bundle.provenance, bundle.dataset_config)


@pytest.mark.parametrize("column,values", [
    ("source_row_id", [0, 0, 3, 4]), ("source_row_id", [0, 3, 1, 4]),
    ("source_row_id", [-1, 1, 3, 4]), ("source_row_id", [0.0, 1.0, 3.0, 4.0]),
    ("source_row_id", [True, 1, 3, 4]), ("source_row_id", [0, 1, 3, 5]),
    ("label_id", [0, 0, 2, 1]), ("label_id", [1.0, 0.0, 2.0, 1.0]),
    ("label_id", [True, 0, 2, 1]), ("label_id", [1, None, 2, 1]),
    ("sentiment", ["neutral", "unknown", "positive", "neutral"]),
    ("sentiment", ["neutral", None, "positive", "neutral"]),
    ("text", ["NA", None, "123", "NA"]), ("text_clean", ["na", "  ", "123", "na"]),
    ("text_clean", ["na", "nan", "123", "different"]),
])
def test_malformed_values_fail_without_silent_cleanup(bundle: InputBundle, column: str, values: list[object]) -> None:
    frame = bundle.frame.copy(deep=True)
    frame[column] = pd.Series(values)
    frame.to_csv(bundle.csv, index=False)
    refresh_checksum(bundle)
    before = bundle.csv.read_bytes()
    with pytest.raises(ValueError):
        load_processed_news(bundle.csv, bundle.provenance, bundle.dataset_config)
    assert bundle.csv.read_bytes() == before


@pytest.mark.parametrize("clean_only", [False, True])
def test_conflicts_keep_full_report(bundle: InputBundle, clean_only: bool) -> None:
    frame = bundle.frame.copy(deep=True)
    frame.loc[1, "text_clean"] = "na"
    if not clean_only:
        frame.loc[1, "text"] = "NA"
    frame.to_csv(bundle.csv, index=False)
    refresh_checksum(bundle)
    with pytest.raises(DuplicateLabelConflictError) as error:
        load_processed_news(bundle.csv, bundle.provenance, bundle.dataset_config)
    assert error.value.report.has_conflicts


def test_missing_class_and_nonstring_table_values(bundle: InputBundle) -> None:
    with pytest.raises(ValueError, match="All three classes"):
        validate_processed_news(bundle.frame.iloc[[0, 2, 3]])
    frame = bundle.frame.copy(deep=True)
    frame["text"] = pd.Series(["NA", 123, "123", "NA"])
    with pytest.raises(ValueError, match="nonblank strings"):
        validate_processed_news(frame)


def test_import_and_load_are_offline_without_fit_or_cleanup(bundle: InputBundle) -> None:
    code = '''
import builtins
from pathlib import Path
import socket
import sys
from unittest.mock import patch
import pandas
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'sklearn', 'nltk'}:
        raise AssertionError('No ML/NLTK import during input preparation')
    return original_import(name, *args, **kwargs)
with patch.object(builtins, '__import__', side_effect=guarded_import), \
     patch.object(Path, 'read_bytes', side_effect=AssertionError('No reads at import')), \
     patch.object(Path, 'read_text', side_effect=AssertionError('No reads at import')), \
     patch.object(pandas, 'read_csv', side_effect=AssertionError('No CSV read at import')):
    from finnews_sentiment.data.load_processed import load_processed_news
    from finnews_sentiment.data.baseline_config import load_baseline_config
with patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden')), \
     patch('finnews_sentiment.features.preprocess.preprocess_text', side_effect=AssertionError('No repeat cleanup')):
    config = load_baseline_config(Path(sys.argv[2]))
    frame = load_processed_news(*(Path(p) for p in sys.argv[3:]))
    assert len(frame) == 4 and config.target_col == 'label_id'
'''
    result = subprocess.run([
        sys.executable, "-c", code, str(ROOT / "src"), str(ROOT / "configs/train_baseline.json"),
        str(bundle.csv), str(bundle.provenance), str(bundle.dataset_config),
    ], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_checked_in_config_is_typed_and_matches_baseline() -> None:
    config = load_baseline_config(ROOT / "configs/train_baseline.json")
    assert config.input_path == Path("data/processed/financial_phrasebank_75agree.csv")
    assert config.provenance_path == Path("reports/day02/provenance.json")
    assert (config.text_col, config.target_col, config.random_state) == ("text_clean", "label_id", 42)
    assert config.test_size == config.validation_size == 0.2
    assert config.tfidf["ngram_range"] == (1, 2)
    assert config.tfidf["max_features"] == 5000
    assert config.logreg["l1_ratio"] == 0.0 and config.logreg["max_iter"] == 1000
    assert config.dummy["strategy"] == "most_frequent"


@pytest.mark.parametrize("section,key,value", [
    ("", "input_path", "/absolute.csv"), ("", "input_path", "../escape.csv"),
    ("", "input_path", ""), ("", "text_col", "text"), ("", "target_col", "sentiment"),
    ("", "test_size", 0), ("", "test_size", 1), ("", "validation_size", float("nan")),
    ("", "random_state", True), ("", "random_state", -1), ("", "random_state", 2**32),
    ("tfidf", "max_features", 0), ("tfidf", "max_features", 3.5), ("tfidf", "max_df", 1.2),
    ("tfidf", "ngram_range", [2, 1]), ("tfidf", "ngram_range", [1, True]),
    ("tfidf", "ngram_range", [1]), ("tfidf", "lowercase", 0), ("tfidf", "stop_words", "english"),
    ("logreg", "C", float("inf")), ("logreg", "max_iter", 0),
    ("logreg", "random_state", 43), ("logreg", "l1_ratio", 1.0),
    ("dummy", "strategy", "stratified"),
])
def test_invalid_config_values(tmp_path: Path, section: str, key: str, value: object) -> None:
    payload = json.loads((ROOT / "configs/train_baseline.json").read_text())
    target = payload[section] if section else payload
    target[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_baseline_config(path)


@pytest.mark.parametrize("problem", ["missing", "extra", "not_object", "nested_not_object"])
def test_invalid_config_schema(tmp_path: Path, problem: str) -> None:
    payload = json.loads((ROOT / "configs/train_baseline.json").read_text())
    if problem == "missing":
        del payload["target_col"]
    elif problem == "extra":
        payload["model_path"] = "legacy.joblib"
    elif problem == "not_object":
        payload = []
    else:
        payload["tfidf"] = []
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_baseline_config(path)
