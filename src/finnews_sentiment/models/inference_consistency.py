"""Read-only validation parity and isolated-process inference checks for Day06."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Literal, TypedDict


STANDALONE_TEXTS = (
    "Aster Components announced revenue of €7.4 million.",
    "<b>No</b> change in the company's annual dividend was announced.",
)


class PredictionPayload(TypedDict):
    text: str
    text_clean: str
    pred_label_id: int
    pred_label: str


class StandaloneEvidence(TypedDict):
    status: Literal["passed"]
    import_without_artifact_io: bool
    training_guard_checked: bool
    file_guard_checked: bool
    network_guard_checked: bool
    state_sha256: str
    runtime_versions: dict[str, str]
    predictions: list[PredictionPayload]


class ParityEvidence(TypedDict):
    status: Literal["exact_match"]
    n_train: int
    n_validation: int
    reference_sha256: str
    pipeline_sha256: str
    public_api_sha256: str


class ConsistencyPayload(TypedDict):
    policy: Literal["day06-consistency-v1"]
    candidate_id: str
    input_checksums: dict[str, str]
    source_checksums: dict[str, str]
    runtime_versions: dict[str, str]
    parity: ParityEvidence
    state_before_sha256: str
    state_after_sha256: str
    serialization_evidence: Literal["D6-03 manifest state checksum and recorded exact validation replay"]
    standalone: StandaloneEvidence


@dataclass(frozen=True)
class ConsistencyRun:
    report_path: Path
    report: ConsistencyPayload


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _standalone_worker(directory: Path) -> None:
    """Internal subprocess entry: allow only copied source/artifacts and runtime reads.

    This audit guard tests access boundaries; it is not an OS security sandbox.
    Nothing is read from the original project's dataset/configs/reports.
    """
    import sysconfig

    workspace = directory.resolve()
    artifact_dir = workspace / "artifact"
    allowed = {workspace, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    allowed.update(Path(path).resolve() for path in sysconfig.get_paths().values())
    # Numerical/runtime imports may inspect shared libraries, timezone or procfs.
    allowed.update(Path(path) for path in ("/usr", "/lib", "/lib64", "/etc", "/proc", "/dev"))
    forbidden_dirs = [workspace / name for name in ("data", "configs", "reports")]
    importing = True

    def guard(event: str, args: tuple[object, ...]) -> None:
        if event in ("socket.connect", "socket.getaddrinfo", "socket.bind", "subprocess.Popen", "os.system"):
            raise PermissionError(f"Standalone forbidden operation: {event}")
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        flags = args[2]
        if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError(f"Standalone writes forbidden: {path}")
        if importing and path.is_relative_to(artifact_dir):
            raise PermissionError("Artifact I/O during import")
        if any(path.is_relative_to(base) for base in forbidden_dirs) or not any(path.is_relative_to(base) for base in allowed):
            raise PermissionError(f"Standalone read forbidden: {path}")

    sys.addaudithook(guard)
    from unittest.mock import patch
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Standalone training/reference/serialization is forbidden")

    from contextlib import ExitStack
    with ExitStack() as stack:
        for cls in (Pipeline, TfidfVectorizer, LogisticRegression, LinearSVC):
            for name in ("fit", "fit_transform", "partial_fit"):
                stack.enter_context(patch.object(cls, name, forbidden, create=True))
        from finnews_sentiment.models import inference_artifact as artifact
        from finnews_sentiment.models.predict import predict_texts

        importing = False
        for name in ("load_inference_reference", "build_inference_bundle", "prepare_inference_artifact", "save_bundle"):
            stack.enter_context(patch.object(artifact, name, forbidden))
        stack.enter_context(patch("joblib.dump", forbidden))
        # Probe installed guards without reading an actual training file or fitting.
        for probe in (lambda: Pipeline.fit(Pipeline([]), [], []),
                      lambda: (workspace / "data/forbidden.csv").read_bytes(),
                      lambda: (workspace.parent / "unavailable_training_root/data/forbidden.csv").read_bytes()):
            try:
                probe()
            except (AssertionError, PermissionError):
                pass
            else:
                raise AssertionError("Standalone guard probe unexpectedly succeeded")
        try:
            sys.audit("socket.connect", None, None)
        except PermissionError:
            pass
        else:
            raise AssertionError("Network guard probe unexpectedly succeeded")
        bundle = artifact.load_bundle(artifact_dir / artifact.BUNDLE_NAME)
        state = artifact._state_sha256(bundle)
        results = predict_texts(bundle, STANDALONE_TEXTS)
        if artifact._state_sha256(bundle) != state:
            raise ValueError("Standalone inference mutated fitted state")
        evidence: StandaloneEvidence = {
            "status": "passed", "import_without_artifact_io": True,
            "training_guard_checked": True, "file_guard_checked": True, "network_guard_checked": True,
            "state_sha256": state, "runtime_versions": artifact._inference_versions(),
            "predictions": [PredictionPayload(text=item.text, text_clean=item.text_clean,
                                               pred_label_id=item.pred_label_id, pred_label=item.pred_label) for item in results],
        }
        print(_json_bytes(evidence).decode("utf-8"), end="")


def _check_standalone(path: Path, expected: StandaloneEvidence, source_root: Path) -> StandaloneEvidence:
    """Copy source and model pair, then compare independently produced evidence."""
    from finnews_sentiment.models.inference_artifact import MANIFEST_NAME
    from finnews_sentiment.data.load_processed import _same

    with tempfile.TemporaryDirectory(prefix="day06-standalone-") as temporary:
        workspace = Path(temporary)
        shutil.copytree(source_root / "finnews_sentiment", workspace / "src/finnews_sentiment",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (workspace / "artifact").mkdir()
        for source in (path, path.with_name(MANIFEST_NAME)):
            shutil.copyfile(source, workspace / "artifact" / source.name)
        code = (
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from finnews_sentiment.models.inference_consistency import _standalone_worker; "
            "_standalone_worker(Path(sys.argv[2]))"
        )
        try:
            result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(workspace / "src"), str(workspace)],
                                    cwd=workspace, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired as error:
            raise ValueError("Standalone inference timed out") from error
        if result.returncode != 0:
            raise ValueError(f"Standalone inference failed: {result.stderr.strip()}")
        try:
            payload: object = json.loads(result.stdout)
        except ValueError as error:
            raise ValueError("Standalone process returned invalid JSON evidence") from error
        _same(payload, expected, "standalone evidence/predictions")
        return expected


def check_inference_consistency(root: Path) -> ConsistencyRun:
    """Check all train/validation cleaning, frozen replay, state and subprocess parity.

    Writes only consistency.json after every check passes; no fit, test model
    calls, serialization or artifact repair. Identical inputs give identical JSON.
    """
    import numpy as np
    from finnews_sentiment.data.load_data import CLASS_ORDER
    from finnews_sentiment.data.load_processed import _same
    from finnews_sentiment.features.preprocess import preprocess_text
    from finnews_sentiment.models.evaluate import _label_ids
    from finnews_sentiment.models.inference_artifact import (
        BUNDLE_NAME, MANIFEST_NAME, _check_build_inputs, _check_bundle_reference,
        _inference_versions, _state_sha256, load_bundle, load_inference_reference,
    )
    from finnews_sentiment.models.predict import predict_texts

    root = root.resolve()
    path = root / "models/day06" / BUNDLE_NAME
    pair_hashes = {str(file.relative_to(root)): _sha(file.read_bytes())
                   for file in (path, path.with_name(MANIFEST_NAME))}
    source_root = Path(__file__).resolve().parents[2]
    source_hashes = {str(file.relative_to(source_root.parent)): _sha(file.read_bytes())
                     for file in sorted((source_root / "finnews_sentiment").rglob("*.py"))}
    reference = load_inference_reference(root)
    bundle = load_bundle(path)
    _check_bundle_reference(bundle, reference)
    state = _state_sha256(bundle)
    metadata = _json_bytes(asdict(bundle.metadata))

    def unchanged_state() -> None:
        if _state_sha256(bundle) != state or _json_bytes(asdict(bundle.metadata)) != metadata:
            raise ValueError("Consistency check detected fitted state/metadata mutation")

    for split, frame, expected_ids in (
        ("train", reference.train, reference.contract.train_source_row_ids),
        ("validation", reference.validation, reference.contract.validation_source_row_ids),
    ):
        _same(frame["source_row_id"].tolist(), list(expected_ids), f"{split} IDs/order")
        for source_id, raw, clean in zip(frame["source_row_id"], frame["text"], frame["text_clean"], strict=True):
            if preprocess_text(raw) != clean:
                raise ValueError(f"Raw/clean mismatch: {split}, source_row_id={source_id}")
    validation = reference.validation
    saved = reference.validation_predictions
    _same(saved["source_row_id"].tolist(), list(reference.contract.validation_source_row_ids), "validation reference IDs/order")
    _same(saved["true_label_id"].tolist(), validation["label_id"].tolist(), "validation reference truth")
    reference_ids = _label_ids(saved["pred_label_id"].to_numpy(), "reference predictions")
    direct = _label_ids(np.asarray(bundle.pipeline.predict(validation["text_clean"].tolist())), "loaded Pipeline predictions")
    unchanged_state()
    api = predict_texts(bundle, validation["text"].tolist())
    unchanged_state()
    _same([item.text for item in api], validation["text"].tolist(), "API raw text/order")
    _same([item.text_clean for item in api], validation["text_clean"].tolist(), "API clean text/order")
    public_ids = _label_ids([item.pred_label_id for item in api], "public API predictions")
    _same([item.pred_label for item in api], [CLASS_ORDER[int(label)] for label in public_ids], "API label mapping")
    if not np.array_equal(reference_ids, direct) or not np.array_equal(reference_ids, public_ids):
        raise ValueError("Validation reference/Pipeline/public API prediction mismatch")
    def aligned_hash(labels: np.ndarray) -> str:
        return _sha(_json_bytes({"source_row_ids": saved["source_row_id"].tolist(),
                                 "true_label_ids": saved["true_label_id"].tolist(), "pred_label_ids": labels.tolist()}))
    standalone_results = predict_texts(bundle, STANDALONE_TEXTS)
    unchanged_state()
    expected: StandaloneEvidence = {
        "status": "passed", "import_without_artifact_io": True,
        "training_guard_checked": True, "file_guard_checked": True, "network_guard_checked": True,
        "state_sha256": state, "runtime_versions": _inference_versions(),
        "predictions": [PredictionPayload(text=item.text, text_clean=item.text_clean,
                                           pred_label_id=item.pred_label_id, pred_label=item.pred_label) for item in standalone_results],
    }
    standalone = _check_standalone(path, expected, source_root)
    unchanged_state()
    _check_build_inputs(reference)
    for base, hashes in ((root, pair_hashes), (source_root.parent, source_hashes)):
        for name, digest in hashes.items():
            if _sha((base / name).read_bytes()) != digest:
                raise ValueError(f"Input/source/artifact changed during consistency check: {base / name}")
    report: ConsistencyPayload = {
        "policy": "day06-consistency-v1", "candidate_id": reference.contract.candidate_id,
        "input_checksums": {**reference.contract.input_checksums, **pair_hashes},
        "source_checksums": source_hashes, "runtime_versions": reference.contract.runtime_versions,
        "parity": {"status": "exact_match", "n_train": len(reference.train), "n_validation": len(validation),
                   "reference_sha256": aligned_hash(reference_ids), "pipeline_sha256": aligned_hash(direct),
                   "public_api_sha256": aligned_hash(public_ids)},
        "state_before_sha256": state, "state_after_sha256": _state_sha256(bundle),
        "serialization_evidence": "D6-03 manifest state checksum and recorded exact validation replay",
        "standalone": standalone,
    }
    report_path = root / "reports/day06/consistency.json"
    content = _json_bytes(report)
    if not report_path.is_file() or report_path.read_bytes() != content:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=report_path.parent, prefix="consistency-", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
        try:
            temporary_path.replace(report_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return ConsistencyRun(report_path, report)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    result = check_inference_consistency(args.root)
    print(f"Validation parity: {result.report['parity']['status']}; N={result.report['parity']['n_validation']}")
    print(f"Standalone: {result.report['standalone']['status']}; report: {result.report_path}")


if __name__ == "__main__":
    main()
