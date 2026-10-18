"""Check Python sources and notebook cells with the project's Pyright config."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_type_check(command: list[str], root: Path, *, minimum_files: int) -> int:
    """Fail closed when Pyright silently skips the requested source files."""
    result = subprocess.run(
        [*command, "--outputjson"], cwd=root, check=False, capture_output=True, text=True,
    )
    try:
        report = json.loads(result.stdout)
        summary = report["summary"]
        analyzed = summary["filesAnalyzed"]
        for diagnostic in report["generalDiagnostics"]:
            location = diagnostic["range"]["start"]
            print(
                f"{diagnostic['file']}:{location['line'] + 1}:{location['character'] + 1}: "
                f"{diagnostic['severity']}: {diagnostic['message']}",
            )
        print(
            f"Analyzed {analyzed} files: {summary['errorCount']} errors, "
            f"{summary['warningCount']} warnings, {summary['informationCount']} informations",
        )
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        print(f"Cannot verify Pyright coverage: {error}\n{result.stdout}\n{result.stderr}", file=sys.stderr)
        return 2
    if not isinstance(analyzed, int) or analyzed < minimum_files:
        print(
            f"Incomplete Pyright coverage: expected at least {minimum_files} files, got {analyzed}. "
            "Check excluded/hidden directories.",
            file=sys.stderr,
        )
        return 2
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    # Pyright skips hidden directories even for explicitly supplied file paths.
    with tempfile.TemporaryDirectory(prefix="notebook-types-", dir=root) as directory:
        exports: list[str] = []
        cell_exports: list[str] = []
        for index, path in enumerate(sorted((root / "notebooks").rglob("*.ipynb"))):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            sources = []
            previous_module: str | None = None
            for number, cell in enumerate(notebook["cells"], start=1):
                if cell["cell_type"] == "code":
                    source = cell["source"]
                    text = "".join(source) if isinstance(source, list) else source
                    sources.append(f"# {path.relative_to(root)} — cell {number}\n{text}\n")
                    # Separate modules preserve declared variable types but do not
                    # propagate a preceding cell's temporary isinstance narrowing.
                    # This supplements the sequential export, not a Pylance emulator.
                    module_name = f"nb{index}_cell_{number}"
                    cell_output = Path(directory) / f"{module_name}.py"
                    imports = f"from {previous_module} import *\n" if previous_module else ""
                    cell_output.write_text(
                        f"# {path.relative_to(root)} — cell {number}\n{imports}{text}\n",
                        encoding="utf-8",
                    )
                    cell_exports.append(str(cell_output))
                    previous_module = module_name
            output = Path(directory) / f"{index}_{path.stem}.py"
            output.write_text("\n".join(sources), encoding="utf-8")
            exports.append(str(output))
        command = [sys.executable, "-m", "pyright", "--project", str(root / "pyrightconfig.json")]
        sources_result = run_type_check(command, root, minimum_files=1)
        if not exports:
            return sources_result
        print(f"Notebook checks: {len(exports)} sequential exports, {len(cell_exports)} cell-boundary modules")
        all_exports = [*exports, *cell_exports]
        notebook_result = run_type_check([*command, *all_exports], root, minimum_files=len(all_exports))
        return sources_result or notebook_result


if __name__ == "__main__":
    raise SystemExit(main())
