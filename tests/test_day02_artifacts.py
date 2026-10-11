"""Offline output/provenance checks using a small local archive fixture."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.data.prepare_day02 import prepare_day02
from finnews_sentiment.data.prepare_phrasebank import parse_phrasebank
from finnews_sentiment.features.labels_duplicates import DuplicateLabelConflictError


def make_source(root: Path, content: str) -> None:
    archive_path = root / "data/raw/FinancialPhraseBank-v1.0.zip"
    archive_path.parent.mkdir(parents=True)
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("Sentences.txt", content.encode("iso-8859-1"))
    config = {
        "dataset": "fixture", "revision": "fixture-revision", "url": "https://example.invalid/no-download",
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "subset": "fixture", "member": "Sentences.txt", "encoding": "iso-8859-1",
        "license": "fixture-license", "citation": "fixture-author", "expected_rows": len(content.splitlines()),
        "class_order": ["negative", "neutral", "positive"],
    }
    (root / "configs").mkdir()
    (root / "configs/dataset.json").write_text(json.dumps(config), encoding="utf-8")
    canonical = root / "data/interim/financial_phrasebank_75agree.csv"
    canonical.parent.mkdir(parents=True)
    parse_phrasebank(content).to_csv(canonical, index=False, lineterminator="\n")


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    make_source(tmp_path, 'NA@neutral\nnan@negative\n123@positive\n<b></b>@neutral\n"£2, NOT -5%"@negative\nNA@neutral\n')
    return tmp_path


def test_roundtrip_reports_and_repeatable_bytes(source_root: Path) -> None:
    source_files = [source_root / "data/raw/FinancialPhraseBank-v1.0.zip", source_root / "data/interim/financial_phrasebank_75agree.csv"]
    before = [path.read_bytes() for path in source_files]
    artifacts = prepare_day02(source_root)
    saved = {key: path.read_bytes() for key, path in asdict(artifacts).items()}
    csv = pd.read_csv(artifacts.processed_csv, keep_default_na=False, dtype={"text": "str", "text_clean": "str"})
    assert csv["text"].tolist() == ["NA", "nan", "123", '"£2, NOT -5%"', "NA"]
    assert csv["source_row_id"].tolist() == [0, 1, 2, 4, 5]
    assert csv["label_id"].tolist() == [1, 0, 2, 0, 1]
    report = json.loads(artifacts.preprocessing_report.read_text())
    assert (report["n_input"], report["n_retained"], report["n_excluded"]) == (6, 5, 1)
    assert report["excluded_rows"] == [{"source_row_id": 3, "reasons": ["empty_after_cleaning"]}]
    duplicate = json.loads(artifacts.duplicate_report.read_text())
    assert duplicate["raw_duplicate_rows"] == duplicate["clean_duplicate_rows"] == 1
    assert duplicate["raw_groups"][0]["source_row_ids"] == [0, 5]
    provenance = json.loads(artifacts.provenance.read_text())
    assert provenance["input"]["sha256"] == hashlib.sha256(before[1]).hexdigest()
    assert provenance["output"]["sha256"] == hashlib.sha256(artifacts.processed_csv.read_bytes()).hexdigest()
    assert provenance["label_mapping"] == {"negative": 0, "neutral": 1, "positive": 2}
    assert provenance["versions"]["pandas"] == pd.__version__
    assert provenance["source"]["revision"] == "fixture-revision"
    assert str(source_root) not in artifacts.provenance.read_text()
    repeated = prepare_day02(source_root)
    assert saved == {key: path.read_bytes() for key, path in asdict(repeated).items()}
    assert before == [path.read_bytes() for path in source_files]


@pytest.mark.parametrize("problem", ["archive", "canonical", "class_order", "missing_archive"])
def test_bad_source_does_not_overwrite_outputs(source_root: Path, problem: str) -> None:
    artifacts = prepare_day02(source_root)
    saved = {key: path.read_bytes() for key, path in asdict(artifacts).items()}
    archive = source_root / "data/raw/FinancialPhraseBank-v1.0.zip"
    if problem == "archive":
        archive.write_bytes(b"tampered")
    elif problem == "missing_archive":
        archive.unlink()
    elif problem == "canonical":
        canonical = source_root / "data/interim/financial_phrasebank_75agree.csv"
        canonical.write_text(canonical.read_text().replace("123", "124"))
    else:
        path = source_root / "configs/dataset.json"
        config = json.loads(path.read_text())
        config["class_order"] = ["positive", "neutral", "negative"]
        path.write_text(json.dumps(config))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_day02(source_root)
    assert saved == {key: path.read_bytes() for key, path in asdict(artifacts).items()}


def test_conflicts_block_all_artifacts(tmp_path: Path) -> None:
    make_source(tmp_path, "Profit@negative\nprofit@positive\n")
    with pytest.raises(DuplicateLabelConflictError):
        prepare_day02(tmp_path)
    assert not (tmp_path / "data/processed").exists()
    assert not (tmp_path / "reports/day02").exists()


def test_cli_is_offline_and_uses_explicit_root(source_root: Path) -> None:
    code = '''
import runpy
import socket
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
sys.argv = ["prepare_day02", "--root", sys.argv[2]]
with patch.object(socket, "socket", side_effect=AssertionError("Network forbidden")):
    runpy.run_module("finnews_sentiment.data.prepare_day02", run_name="__main__")
'''
    result = subprocess.run([sys.executable, "-c", code, str(ROOT / "src"), str(source_root)],
                            capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "processed_csv:" in result.stdout
    assert (source_root / "reports/day02/provenance.json").is_file()
