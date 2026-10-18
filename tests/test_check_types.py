"""The type gate must analyze notebook exports and fail on skipped files."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("source", "exclude_exports", "expected_code", "expected_message"),
    [
        ('answer: int = 42', False, 0, '0 errors'),
        ('answer: int = "wrong"', False, 1, 'not assignable'),
        ('answer: int = 42', True, 2, 'Incomplete Pyright coverage'),
        (
            [
                'def payload() -> dict[str, object]:\n'
                '    return {"selected": {"per_class": []}}\n'
                'metrics = payload()["selected"]\n'
                'if not isinstance(metrics, dict):\n'
                '    raise TypeError()\n',
                'rows = metrics["per_class"]',
            ],
            False, 1, '__getitem__',
        ),
        (
            [
                'from typing import TypedDict\n'
                'class Metrics(TypedDict):\n'
                '    per_class: list[int]\n'
                'def payload() -> Metrics:\n'
                '    return {"per_class": []}\n'
                'metrics = payload()\n',
                'rows = metrics["per_class"]',
            ],
            False, 0, '0 errors',
        ),
    ],
)
def test_gate_checks_notebook_types_and_coverage(
    tmp_path: Path, source: str | list[str], exclude_exports: bool,
    expected_code: int, expected_message: str,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "notebooks").mkdir()
    shutil.copyfile(ROOT / "scripts/check_types.py", tmp_path / "scripts/check_types.py")
    config: dict[str, object] = {"include": ["scripts"], "typeCheckingMode": "standard"}
    if exclude_exports:
        config["exclude"] = ["notebook-types-*/**"]
    (tmp_path / "pyrightconfig.json").write_text(json.dumps(config), encoding="utf-8")
    sources = [source] if isinstance(source, str) else source
    notebook = {"cells": [{"cell_type": "code", "source": [text]} for text in sources]}
    (tmp_path / "notebooks/probe.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(tmp_path / "scripts/check_types.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == expected_code, result.stdout + result.stderr
    assert expected_message in result.stdout + result.stderr
    assert not list(tmp_path.glob("notebook-types-*"))
