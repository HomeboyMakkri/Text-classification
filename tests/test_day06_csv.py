"""CSV adapter preserves parsed fields/order and publishes only complete outputs."""

import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.models import inference_artifact as artifact
from finnews_sentiment.models import predict
from test_day06_artifact import saved_pair, completed_reference, day04_reference


def _forbid(*args: object, **kwargs: object) -> None:
    raise AssertionError("CSV inference must not fit, access training references or use network")


@pytest.fixture
def model(saved_pair: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Pipeline, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "fit_transform", _forbid)
    monkeypatch.setattr(artifact, "load_inference_reference", _forbid)
    monkeypatch.setattr("socket.create_connection", _forbid)
    monkeypatch.setattr("urllib.request.urlopen", _forbid)
    return saved_pair


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(header)
        writer.writerows(rows)


def _read(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream, strict=True))


def test_csv_api_equivalence_and_string_roundtrip(model: Path, tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    output = tmp_path / "nested/output.csv"
    header = ["request_id", "text", "note"]
    rows = [
        ["0001", '<b>No</b> growth,\nprofit is NOT down -5% &amp; costs £2.50.', 'NA'],
        ["0002", "nan", 'quoted "note", €\r\nnext line'],
        ["0001", '<b>No</b> growth,\nprofit is NOT down -5% &amp; costs £2.50.', 'NA'],
        ["0003", "NA", "a lone\rreturn"],
        ["0004", "123.45 €", ""],
    ]
    _write(source, header, rows)
    before = source.read_bytes()
    pair_before = {path: path.read_bytes() for path in model.parent.iterdir()}
    expected = predict.predict_texts(artifact.load_bundle(model), [row[1] for row in rows])
    predict.predict_for_file(model, source, output)
    actual = _read(output)
    assert actual[0] == [*header, *predict.PREDICTION_COLUMNS]
    assert [row[:len(header)] for row in actual[1:]] == rows
    assert [row[len(header):] for row in actual[1:]] == [
        [item.text_clean, str(item.pred_label_id), item.pred_label] for item in expected
    ]
    assert actual[1] == actual[3]
    assert source.read_bytes() == before
    assert all(path.read_bytes() == content for path, content in pair_before.items())
    assert not list(output.parent.glob(".inference-*"))


@pytest.mark.parametrize("content,pattern", [
    (b"", "empty"), (b"\n", "empty"), (b"text\n", "at least one"),
    (b"headline\nhello\n", "text column"), (b"text,text\na,b\n", "unique"),
    (b"text,extra,extra\na,b,c\n", "unique"), (b"text,\na,b\n", "blank"),
    (b"text,extra\nhello\n", "fields"), (b"text\nhello,extra\n", "fields"),
    (b"text\n\n", "fields"), (b'text\n"unclosed\n', "unexpected end"),
    (b"text\n\xff\n", "utf-8"),
])
def test_structural_input_errors_precede_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes, pattern: str,
) -> None:
    source = tmp_path / "bad.csv"
    source.write_bytes(content)
    monkeypatch.setattr(predict, "load_bundle", _forbid)
    with pytest.raises(ValueError, match=pattern):
        predict.predict_for_file(tmp_path / "missing.joblib", source, tmp_path / "absent/output.csv")
    assert source.read_bytes() == content
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("column", predict.PREDICTION_COLUMNS)
def test_prediction_column_collision(model: Path, tmp_path: Path, column: str) -> None:
    source = tmp_path / "input.csv"
    _write(source, ["text", column], [["A new financial sentence.", "old"]])
    with pytest.raises(ValueError, match="already contains prediction columns"):
        predict.predict_for_file(model, source, tmp_path / "output.csv")
    assert not (tmp_path / "output.csv").exists()


@pytest.mark.parametrize("text", ["", " \t\n", "<b></b>", "&nbsp;"])
def test_missing_or_empty_text_rejects_whole_csv_before_predict(model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    source = tmp_path / "input.csv"
    _write(source, ["text"], [["A valid first sentence."], [text]])
    monkeypatch.setattr(Pipeline, "predict", _forbid)
    with pytest.raises(ValueError, match=r"texts\[1\]"):
        predict.predict_for_file(model, source, tmp_path / "output.csv")
    assert not (tmp_path / "output.csv").exists()


@pytest.mark.parametrize("kind", ["same", "symlink", "hardlink", "existing", "model", "manifest", "dangling"])
def test_output_never_overwrites_existing_paths(model: Path, tmp_path: Path, kind: str) -> None:
    source = tmp_path / "input.csv"
    _write(source, ["text"], [["New unlabeled text."]])
    output = tmp_path / "output.csv"
    if kind == "same":
        output = source
    elif kind == "symlink":
        output.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, output)
    elif kind == "existing":
        output.write_bytes(b"preserve existing output")
    elif kind == "model":
        output = model
    elif kind == "manifest":
        output = model.with_name(artifact.MANIFEST_NAME)
    else:
        output.symlink_to(tmp_path / "missing-target")
    before = {path: path.read_bytes() for path in (source, model, model.with_name(artifact.MANIFEST_NAME))}
    output_before = output.read_bytes() if output.exists() else None
    with pytest.raises((ValueError, FileExistsError)):
        predict.predict_for_file(model, source, output)
    assert all(path.read_bytes() == content for path, content in before.items())
    if output_before is not None:
        assert output.read_bytes() == output_before
    assert not (tmp_path / "missing-target").exists()


def test_publication_race_does_not_replace_another_output(model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    _write(source, ["text"], [["New text"]])
    original = predict.predict_texts

    def concurrent(bundle, texts):
        result = original(bundle, texts)
        output.write_bytes(b"another writer")
        return result

    monkeypatch.setattr(predict, "predict_texts", concurrent)
    with pytest.raises(FileExistsError):
        predict.predict_for_file(model, source, output)
    assert output.read_bytes() == b"another writer"
    assert not list(tmp_path.glob(".inference-*"))


def test_input_change_during_inference_blocks_publication(model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    _write(source, ["text"], [["New text"]])
    original = predict.predict_texts

    def mutate(bundle, texts):
        result = original(bundle, texts)
        source.write_bytes(source.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(predict, "predict_texts", mutate)
    with pytest.raises(ValueError, match="Input CSV changed"):
        predict.predict_for_file(model, source, output)
    assert not output.exists()


def test_publication_failure_leaves_no_partial_output(model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    _write(source, ["text"], [["New text"]])

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("fixture publication failure")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError, match="publication failure"):
        predict.predict_for_file(model, source, output)
    assert not output.exists() and not list(tmp_path.glob(".inference-*"))


def test_cli_matches_api_without_fit_or_network(model: Path, tmp_path: Path) -> None:
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    _write(source, ["text", "id"], [["<b>No</b> growth in profit.", "001"], ["NA", "002"]])
    # Fresh process with explicit training/network/reference guards; invoke the real CLI.
    code = """
import runpy
from unittest.mock import patch
from contextlib import ExitStack
def forbidden(*args, **kwargs):
    raise AssertionError('CSV CLI must not train or access references/network')
with ExitStack() as stack:
    for name in ('sklearn.pipeline.Pipeline.fit', 'sklearn.pipeline.Pipeline.fit_transform',
                 'finnews_sentiment.models.inference_artifact.load_inference_reference',
                 'socket.create_connection', 'urllib.request.urlopen', 'joblib.dump'):
        stack.enter_context(patch(name, forbidden))
    runpy.run_module('finnews_sentiment.models.predict', run_name='__main__')
"""
    command = [str(ROOT / "venv/bin/python"), "-c", code, "--model", str(model), "--input", str(source), "--output", str(output)]
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    result = subprocess.run(command, cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected = predict.predict_texts(artifact.load_bundle(model), [row[0] for row in _read(source)[1:]])
    assert [row[-2:] for row in _read(output)[1:]] == [[str(item.pred_label_id), item.pred_label] for item in expected]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
