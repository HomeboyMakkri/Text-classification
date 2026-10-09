"""Check Python sources and notebook cells with the project's Pyright config."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="notebook-types-", dir=root / ".cache") as directory:
        exports: list[str] = []
        for index, path in enumerate(sorted((root / "notebooks").rglob("*.ipynb"))):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            sources = []
            for number, cell in enumerate(notebook["cells"], start=1):
                if cell["cell_type"] == "code":
                    source = cell["source"]
                    text = "".join(source) if isinstance(source, list) else source
                    sources.append(f"# {path.relative_to(root)} — cell {number}\n{text}\n")
            output = Path(directory) / f"{index}_{path.stem}.py"
            output.write_text("\n".join(sources), encoding="utf-8")
            exports.append(str(output))
        command = [sys.executable, "-m", "pyright", "--project", str(root / "pyrightconfig.json")]
        sources_result = subprocess.run(command, cwd=root, check=False)
        if not exports:
            return sources_result.returncode
        notebook_result = subprocess.run([*command, *exports], cwd=root, check=False)
        return sources_result.returncode or notebook_result.returncode


if __name__ == "__main__":
    (Path(__file__).resolve().parents[1] / ".cache").mkdir(exist_ok=True)
    raise SystemExit(main())
