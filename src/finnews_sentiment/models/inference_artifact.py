"""Frozen train-only build, trusted local serialization and read-only loading."""

from dataclasses import asdict, dataclass, fields
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
import tempfile
from typing import Literal, TypedDict
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from finnews_sentiment.data.baseline_config import _integer, load_baseline_config
from finnews_sentiment.data.experiment_splits import _object, load_day04_inputs
from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.data.load_processed import _hash, _same
from finnews_sentiment.features.labels_duplicates import LABEL_MAPPING
from finnews_sentiment.features import preprocess

BUNDLE_SCHEMA = "day06-inference-v1"
BUNDLE_NAME = "sentiment_bundle.joblib"
MANIFEST_NAME = "manifest.json"
INFERENCE_PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")


class PipelineParameters(TypedDict):
    """JSON-compatible effective parameters, including sklearn defaults."""

    tfidf: dict[str, object]
    classifier: dict[str, object]


@dataclass(frozen=True)
class FrozenInferenceContract:
    """Verified configuration/provenance; does not claim a Day06 model exists."""

    schema_version: Literal["day06-inference-v1"]
    candidate_id: str
    parameters: PipelineParameters
    pipeline_parameters: PipelineParameters
    preprocessing_policy: Literal["minimal-v1"]
    preprocessing_parameters: dict[str, object]
    preprocessing_sha256: str
    class_order: tuple[str, ...]
    label_mapping: dict[str, int]
    fit_policy: Literal["day03_train_only"]
    random_state: int
    text_col: str
    target_col: str
    train_source_row_ids: tuple[int, ...]
    validation_source_row_ids: tuple[int, ...]
    input_checksums: dict[str, str]
    runtime_versions: dict[str, str]


@dataclass(frozen=True)
class InferenceReference:
    """Detached train/validation snapshots only; no test data or estimators.

    DataFrames/dicts remain mutable Python values; frozen prevents rebinding,
    not deep mutation. Callers must not modify these validated snapshots.
    """

    root: Path
    train: pd.DataFrame
    validation: pd.DataFrame
    validation_predictions: pd.DataFrame
    split_summary: pd.DataFrame
    contract: FrozenInferenceContract
    day05_n_features: int
    day05_validation_replay: Literal["exact_match"]


@dataclass(frozen=True)
class BundleMetadata:
    """Successful fresh Day06 build evidence, distinct from cached Day05 replay."""

    contract: FrozenInferenceContract
    n_features: int
    validation_replay: Literal["exact_match"]
    n_train: int
    n_validation: int
    validation_reference_sha256: str


@dataclass(frozen=True)
class InferenceBundle:
    """Fitted TF-IDF/classifier Pipeline and its checked metadata, in memory."""

    pipeline: Pipeline
    metadata: BundleMetadata


def _strings(value: object, name: str) -> dict[str, str]:
    payload = _object(value, name)
    result: dict[str, str] = {}
    for key, item in payload.items():
        if not isinstance(item, str) or not item:
            raise ValueError(f"{name}.{key} must be a nonempty string")
        result[key] = item
    return result


def _parameters(value: object, name: str) -> PipelineParameters:
    # Day05's validated in-memory protocol still has sklearn tuple values.
    # Use its saved JSON representation: ngram_range is a list, dtype a name.
    payload = _object(json.loads(json.dumps(value, allow_nan=False)), name)
    if set(payload) != {"tfidf", "classifier"}:
        raise ValueError(f"{name} must contain tfidf and classifier")
    return {
        "tfidf": _object(payload["tfidf"], f"{name}.tfidf"),
        "classifier": _object(payload["classifier"], f"{name}.classifier"),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inference_reference(root: Path) -> InferenceReference:
    """Verify frozen Day02–Day05 artifacts offline, without model calls/writes.

    Existing readers validate full effective parameters, selection, versions,
    checksums, IDs/labels/order and group isolation. Final-cache validation
    checks saved outputs; it never runs new test predictions. Only train and
    validation are exposed for the future explicit build.
    """
    # Keep preparation-only dependencies out of future standalone load/predict.
    from finnews_sentiment.models.error_analysis import load_validation_analysis
    from finnews_sentiment.models.final_evaluation import FINAL_NAMES, load_final_evaluation

    root = root.resolve()
    directory = root / "reports/day05"
    try:
        # Capture before validation; do not trust freshly changed marker bytes.
        final_hashes = {str((directory / name).relative_to(root)): _sha(directory / name)
                        for name in FINAL_NAMES}
        final = load_final_evaluation(root)
        inputs = load_day04_inputs(root)
        analysis = load_validation_analysis(root)
        config = load_baseline_config(root / "configs/train_baseline.json")
        protocol = final.protocol
        _same(protocol.get("policy"), "day05-frozen-protocol-v1", "protocol.policy")
        _same(protocol.get("candidate_id"), analysis.candidate_id, "protocol.candidate_id")
        _same(protocol.get("preprocessing_policy"), "minimal-v1", "preprocessing_policy")
        _same(preprocess.PREPROCESSING_POLICY, "minimal-v1", "runtime preprocessing_policy")
        _same(protocol.get("fit_policy"), "day03_train_only", "fit_policy")
        _same(protocol.get("class_order"), list(CLASS_ORDER), "class_order")
        _same(protocol.get("label_mapping"), LABEL_MAPPING, "label_mapping")
        _same(protocol.get("random_state"), config.random_state, "random_state")
        _same(protocol.get("text_col"), config.text_col, "text_col")
        _same(protocol.get("target_col"), config.target_col, "target_col")
        train = inputs.splits["train"].copy(deep=True)
        validation = inputs.splits["validation"].copy(deep=True)
        for split, frame in (("train", train), ("validation", validation)):
            _same(protocol.get(f"{split}_source_row_ids"), frame["source_row_id"].tolist(),
                  f"{split}_source_row_ids")
        predictions = analysis.table[["source_row_id", "true_label_id", "pred_label_id"]].copy(deep=True)
        if (predictions["source_row_id"].tolist() != validation["source_row_id"].tolist()
                or predictions["true_label_id"].tolist() != validation["label_id"].tolist()):
            raise ValueError("Day06 validation reference IDs/order/truth mismatch")
        hashes = _strings(protocol.get("input_checksums"), "input_checksums")
        hashes.update(final_hashes)
        # Cross-reader consistency: nothing may change while snapshots are made.
        for name, expected in hashes.items():
            if _sha(root / name) != expected:
                raise ValueError(f"Frozen input changed during Day06 preparation: {root / name}")
        source = preprocess.__file__
        if source is None:
            raise ValueError("Cannot locate features/preprocess.py implementation")
        contract = FrozenInferenceContract(
            schema_version=BUNDLE_SCHEMA, candidate_id=analysis.candidate_id,
            parameters=_parameters(protocol.get("parameters"), "parameters"),
            pipeline_parameters=_parameters(protocol.get("pipeline_parameters"), "pipeline_parameters"),
            preprocessing_policy="minimal-v1",
            preprocessing_parameters=_object(protocol.get("preprocessing_parameters"), "preprocessing_parameters"),
            preprocessing_sha256=_sha(Path(source)), class_order=CLASS_ORDER,
            label_mapping=dict(LABEL_MAPPING), fit_policy="day03_train_only",
            random_state=config.random_state, text_col=config.text_col, target_col=config.target_col,
            train_source_row_ids=tuple(train["source_row_id"].tolist()),
            validation_source_row_ids=tuple(validation["source_row_id"].tolist()),
            input_checksums=hashes, runtime_versions=_strings(protocol.get("runtime_versions"), "runtime_versions"),
        )
        return InferenceReference(
            root, train, validation, predictions, inputs.split_summary.copy(deep=True),
            contract, final.n_features, "exact_match",
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Day06 frozen reference rejected under {root}: {error}. "
            "Inspect Day02–Day05 paths, checksums and frozen contracts; see docs/day05.md "
            "for explicit preparation commands. No automatic rebuild, fit or test evaluation."
        ) from error


def _same_reference(reference: InferenceReference, verified: InferenceReference) -> None:
    """Reject mutation of the mutable values inside a frozen dataclass."""
    if (reference.contract != verified.contract
            or reference.day05_n_features != verified.day05_n_features
            or reference.day05_validation_replay != verified.day05_validation_replay):
        raise ValueError("Day06 reference metadata differs from frozen artifacts")
    for name in ("train", "validation", "validation_predictions", "split_summary"):
        try:
            pd.testing.assert_frame_equal(getattr(reference, name), getattr(verified, name), check_exact=True)
        except AssertionError as error:
            raise ValueError(f"Day06 reference {name} differs from frozen artifacts: {error}") from error


def _check_build_inputs(reference: InferenceReference) -> None:
    """Check the bytes used for preparation again, before fit and after replay."""
    from finnews_sentiment.models.final_evaluation import runtime_versions

    for name, expected in reference.contract.input_checksums.items():
        if _sha(reference.root / name) != expected:
            raise ValueError(f"Frozen input changed during Day06 build: {reference.root / name}")
    source = preprocess.__file__
    if (source is None or _sha(Path(source)) != reference.contract.preprocessing_sha256
            or preprocess.PREPROCESSING_POLICY != reference.contract.preprocessing_policy):
        raise ValueError("Preprocessing implementation/policy changed during Day06 build")
    if runtime_versions() != reference.contract.runtime_versions:
        raise ValueError("Runtime versions changed during Day06 build")


def build_inference_bundle(reference: InferenceReference) -> InferenceBundle:
    """Explicit fit on frozen train and exact replay on validation; never test.

    Revalidate caller snapshots before any model calls. Only detached train
    texts/labels and validation texts reach sklearn. Returns an in-memory
    bundle after fresh replay; does not serialize or write reports.
    """
    from finnews_sentiment.models.evaluate import _label_ids
    from finnews_sentiment.models.run_experiments_day04 import _candidate_parameters, _candidate_pipeline

    try:
        verified = load_inference_reference(reference.root)
        _same_reference(reference, verified)
        contract = verified.contract
        for split, frame in (("train", verified.train), ("validation", verified.validation)):
            for source_id, raw, clean in zip(frame["source_row_id"], frame["text"], frame["text_clean"], strict=True):
                if preprocess.preprocess_text(raw) != clean:
                    raise ValueError(f"Raw/clean preprocessing mismatch: {split}, source_row_id={source_id}")
        config = load_baseline_config(verified.root / "configs/train_baseline.json")
        pipeline, selected_config = _candidate_pipeline(contract.candidate_id, config)
        vectorizer = pipeline.named_steps.get("tfidf")
        classifier = pipeline.named_steps.get("clf")
        if not isinstance(vectorizer, TfidfVectorizer) or classifier is None:
            raise ValueError("Selected Pipeline must contain TF-IDF and classifier")
        tfidf_params = vectorizer.get_params(deep=False)
        tfidf_params["dtype"] = np.dtype(tfidf_params["dtype"]).name
        effective = _parameters({"tfidf": tfidf_params, "classifier": classifier.get_params(deep=False)},
                                "pipeline_parameters")
        _same(effective, contract.pipeline_parameters, "pipeline_parameters")
        selected = _parameters(_candidate_parameters(pipeline, selected_config), "parameters")
        _same(selected, contract.parameters, "parameters")
        _check_build_inputs(verified)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(verified.train[contract.text_col].tolist(),
                         verified.train[contract.target_col].to_numpy())
            n_features = len(vectorizer.get_feature_names_out())
            if n_features != verified.day05_n_features:
                raise ValueError("Day06 feature count differs from frozen Day05 fit policy")
            vocabulary = dict(vectorizer.vocabulary_)
            idf = vectorizer.idf_.copy()
            classes = np.asarray(pipeline.classes_).copy()
            if not np.array_equal(classes, [0, 1, 2]):
                raise ValueError("Fitted classifier classes must be [0, 1, 2]")
            predicted = _label_ids(np.asarray(pipeline.predict(verified.validation[contract.text_col].tolist())),
                                   "Day06 validation replay")
            if not np.array_equal(predicted, verified.validation_predictions["pred_label_id"].to_numpy()):
                raise ValueError("Day06 validation replay mismatch; no bundle returned")
            if (vocabulary != vectorizer.vocabulary_ or not np.array_equal(idf, vectorizer.idf_)
                    or not np.array_equal(classes, pipeline.classes_)):
                raise ValueError("Validation replay changed fitted vocabulary/IDF/classes")
        _same_reference(reference, verified)
        _check_build_inputs(verified)
        return InferenceBundle(pipeline, BundleMetadata(
            contract=contract, n_features=n_features, validation_replay="exact_match",
            n_train=len(verified.train), n_validation=len(verified.validation),
            validation_reference_sha256=contract.input_checksums["reports/day04/validation_predictions.csv"],
        ))
    except (OSError, ValueError, ConvergenceWarning) as error:
        raise ValueError(f"Day06 build stopped before returning a bundle: {error}") from error


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _inference_versions() -> dict[str, str]:
    # Do not import the preparation/report stack or require matplotlib to load.
    return {"python": platform.python_version(), **{name: version(name) for name in INFERENCE_PACKAGES}}


def _metadata_payload(metadata: BundleMetadata) -> dict[str, object]:
    return _object(json.loads(_json_bytes(asdict(metadata))), "bundle metadata")


def _preflight_metadata(value: object) -> dict[str, object]:
    """Validate readable metadata before unpickling; no training data needed."""
    metadata = _object(value, "bundle metadata")
    if set(metadata) != {field.name for field in fields(BundleMetadata)}:
        raise ValueError("Bundle metadata schema mismatch")
    contract = _object(metadata.get("contract"), "frozen contract")
    if set(contract) != {field.name for field in fields(FrozenInferenceContract)}:
        raise ValueError("Frozen contract schema mismatch")
    for name, expected in (("schema_version", BUNDLE_SCHEMA), ("preprocessing_policy", "minimal-v1"),
                           ("fit_policy", "day03_train_only"), ("class_order", list(CLASS_ORDER)),
                           ("label_mapping", LABEL_MAPPING), ("text_col", "text_clean"), ("target_col", "label_id")):
        _same(contract.get(name), expected, name)
    _same(metadata.get("validation_replay"), "exact_match", "validation_replay")
    candidate = contract.get("candidate_id")
    if candidate not in ("baseline", "word_unigram", "linear_svc", "logreg_c_0_1", "logreg_c_1_0", "logreg_c_10_0"):
        raise ValueError(f"Unsupported candidate_id: {candidate!r}")
    _integer(contract.get("random_state"), "random_state", minimum=0)
    _parameters(contract.get("parameters"), "parameters")
    _parameters(contract.get("pipeline_parameters"), "pipeline_parameters")
    _same(contract.get("preprocessing_parameters"), {
        "duplicate_policy": "retain_same_label_block_conflicts", "source_row_id": "zero_based_canonical_position",
    }, "preprocessing_parameters")
    _integer(metadata.get("n_features"), "n_features")
    ids: dict[str, list[int]] = {}
    for split in ("train", "validation"):
        count = _integer(metadata.get(f"n_{split}"), f"n_{split}")
        saved_ids = contract.get(f"{split}_source_row_ids")
        if not isinstance(saved_ids, list) or len(saved_ids) != count:
            raise ValueError(f"{split} IDs/count mismatch")
        parsed = [_integer(item, f"{split} source ID", minimum=0) for item in saved_ids]
        if any(left >= right for left, right in zip(parsed, parsed[1:])):
            raise ValueError(f"{split} IDs must be unique in canonical order")
        ids[split] = parsed
    if not set(ids["train"]).isdisjoint(ids["validation"]):
        raise ValueError("Train/validation IDs overlap")
    hashes = _strings(contract.get("input_checksums"), "input_checksums")
    for name, digest in hashes.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid input checksum path: {name}")
        _hash(digest, name)
    reference_hash = _hash(metadata.get("validation_reference_sha256"), "validation_reference_sha256")
    _same(hashes.get("reports/day04/validation_predictions.csv"), reference_hash, "validation reference checksum")
    saved_versions = _strings(contract.get("runtime_versions"), "runtime_versions")
    current = _inference_versions()
    _same({name: saved_versions.get(name) for name in current}, current, "inference runtime versions")
    source = preprocess.__file__
    if (source is None or preprocess.PREPROCESSING_POLICY != "minimal-v1"
            or _hash(contract.get("preprocessing_sha256"), "preprocessing_sha256") != _sha(Path(source))):
        raise ValueError("Preprocessing implementation/policy mismatch")
    return metadata


def _validate_bundle(value: object) -> InferenceBundle:
    """Check fitted components, dimensions and metadata after deserialize."""
    if (not isinstance(value, InferenceBundle) or not isinstance(value.metadata, BundleMetadata)
            or not isinstance(value.metadata.contract, FrozenInferenceContract)):
        raise ValueError("Expected InferenceBundle with typed metadata/contract")
    bundle = value
    _preflight_metadata(_metadata_payload(bundle.metadata))
    pipeline = bundle.pipeline
    if not isinstance(pipeline, Pipeline) or [name for name, _ in pipeline.steps] != ["tfidf", "clf"]:
        raise ValueError("Bundle requires a tfidf -> clf Pipeline")
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    expected_type = LinearSVC if bundle.metadata.contract.candidate_id == "linear_svc" else LogisticRegression
    if not isinstance(vectorizer, TfidfVectorizer) or not isinstance(classifier, expected_type):
        raise ValueError("Bundle estimator types differ from candidate")
    check_is_fitted(vectorizer, ["vocabulary_", "idf_"])
    check_is_fitted(classifier, ["classes_", "coef_", "intercept_", "n_features_in_"])
    n_features = bundle.metadata.n_features
    vocabulary = vectorizer.vocabulary_
    if (len(vocabulary) != n_features or any(not isinstance(word, str) or not word for word in vocabulary)
            or any(not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)) for index in vocabulary.values())
            or set(vocabulary.values()) != set(range(n_features))):
        raise ValueError("Fitted vocabulary dimensions/indexes mismatch")
    if (not np.array_equal(classifier.classes_, [0, 1, 2])
            or not np.issubdtype(np.asarray(classifier.classes_).dtype, np.integer)):
        raise ValueError("Fitted classes must be integer IDs [0, 1, 2]")
    if (vectorizer.idf_.shape != (n_features,) or classifier.coef_.shape != (3, n_features)
            or classifier.intercept_.shape != (3,) or classifier.n_features_in_ != n_features):
        raise ValueError("Fitted IDF/classifier feature dimensions mismatch")
    if any(not np.isfinite(array).all() for array in (vectorizer.idf_, classifier.coef_, classifier.intercept_)):
        raise ValueError("Fitted state contains non-finite values")
    tfidf_params = vectorizer.get_params(deep=False)
    tfidf_params["dtype"] = np.dtype(tfidf_params["dtype"]).name
    parameters = _parameters({"tfidf": tfidf_params, "classifier": classifier.get_params(deep=False)}, "parameters")
    _same(parameters, bundle.metadata.contract.pipeline_parameters, "effective pipeline parameters")
    _same(parameters["classifier"], bundle.metadata.contract.parameters["classifier"], "selected classifier parameters")
    for name, expected in bundle.metadata.contract.parameters["tfidf"].items():
        _same(parameters["tfidf"].get(name), expected, f"selected tfidf.{name}")
    return bundle


def _state_sha256(bundle: InferenceBundle) -> str:
    vectorizer = bundle.pipeline.named_steps["tfidf"]
    classifier = bundle.pipeline.named_steps["clf"]
    # sklearn may store NumPy integer indices; normalize only the hash payload.
    vocabulary = {word: int(index) for word, index in vectorizer.vocabulary_.items()}
    digest = hashlib.sha256(_json_bytes(vocabulary))
    for value in (vectorizer.idf_, classifier.classes_, classifier.coef_, classifier.intercept_):
        array = np.asarray(value)
        digest.update(_json_bytes({"shape": list(array.shape), "dtype": str(array.dtype)}))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_bundle(path: Path) -> InferenceBundle:
    """Read a trusted local bundle/manifest without fit, predict or dataset I/O."""
    path = path.resolve()
    try:
        manifest_bytes = path.with_name(MANIFEST_NAME).read_bytes()
        manifest = _object(json.loads(manifest_bytes), "manifest")
        if set(manifest) != {"schema_version", "bundle_file", "bundle_sha256", "state_sha256", "metadata"}:
            raise ValueError("Manifest schema mismatch")
        _same(manifest.get("schema_version"), BUNDLE_SCHEMA, "manifest schema_version")
        _same(manifest.get("bundle_file"), path.name, "manifest bundle_file")
        metadata = _preflight_metadata(manifest.get("metadata"))
        expected = _hash(manifest.get("bundle_sha256"), "bundle_sha256")
        state_hash = _hash(manifest.get("state_sha256"), "state_sha256")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Bundle checksum mismatch")
        try:
            value: object = joblib.load(BytesIO(content))
        except Exception as error:
            raise ValueError(f"Cannot deserialize trusted bundle: {error}") from error
        bundle = _validate_bundle(value)
        _same(_metadata_payload(bundle.metadata), metadata, "bundle/manifest metadata agreement")
        if _state_sha256(bundle) != state_hash:
            raise ValueError("Fitted state checksum mismatch")
        if _sha(path) != expected or path.with_name(MANIFEST_NAME).read_bytes() != manifest_bytes:
            raise ValueError("Bundle/manifest changed during load")
        return bundle
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"Cannot load Day06 artifact {path}: {error}. Inspect the pair; no automatic rebuild or overwrite.") from error


def _check_bundle_reference(bundle: InferenceBundle, reference: InferenceReference) -> None:
    _same(_metadata_payload(bundle.metadata)["contract"],
          json.loads(_json_bytes(asdict(reference.contract))), "bundle/frozen reference contract")
    if (bundle.metadata.n_train != len(reference.train) or bundle.metadata.n_validation != len(reference.validation)
            or bundle.metadata.n_features != reference.day05_n_features):
        raise ValueError("Bundle/frozen reference counts mismatch")


def save_bundle(bundle: InferenceBundle, directory: Path, *, reference: InferenceReference) -> Path:
    """Roundtrip/replay in staging, publish bundle then manifest without overwrite."""
    from finnews_sentiment.models.evaluate import _label_ids

    directory = directory.resolve()
    path = directory / BUNDLE_NAME
    verified = load_inference_reference(reference.root)
    _same_reference(reference, verified)
    _validate_bundle(bundle)
    _check_bundle_reference(bundle, verified)
    if directory.exists() and any(directory.iterdir()):
        existing = load_bundle(path)  # Partial or incompatible pair is an error.
        _check_bundle_reference(existing, verified)
        if _state_sha256(existing) != _state_sha256(bundle):
            raise ValueError("Existing artifact has different fitted state; use a new output directory")
        _check_build_inputs(verified)
        return path
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="day06-staging-", dir=directory.parent) as temporary:
        staged = Path(temporary) / BUNDLE_NAME
        joblib.dump(bundle, staged, compress=3)
        manifest = {"schema_version": BUNDLE_SCHEMA, "bundle_file": BUNDLE_NAME,
                    "bundle_sha256": _sha(staged), "state_sha256": _state_sha256(bundle),
                    "metadata": _metadata_payload(bundle.metadata)}
        manifest_bytes = _json_bytes(manifest)
        staged.with_name(MANIFEST_NAME).write_bytes(manifest_bytes)
        loaded = load_bundle(staged)
        before = _state_sha256(loaded)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            predictions = _label_ids(np.asarray(loaded.pipeline.predict(verified.validation[verified.contract.text_col].tolist())),
                                     "roundtrip validation replay")
        if not np.array_equal(predictions, verified.validation_predictions["pred_label_id"].to_numpy()):
            raise ValueError("Roundtrip validation replay mismatch; artifact not published")
        if before != _state_sha256(loaded) or before != _state_sha256(bundle):
            raise ValueError("Roundtrip/replay changed fitted state; artifact not published")
        _same_reference(reference, verified)
        _check_build_inputs(verified)
        directory.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents replacement; crash may leave a partial pair.
        with path.open("xb") as stream:
            stream.write(staged.read_bytes())
        if _sha(path) != manifest["bundle_sha256"]:
            raise ValueError("Published bundle read-back mismatch; inspect partial output")
        with (directory / MANIFEST_NAME).open("xb") as stream:
            stream.write(manifest_bytes)
    load_bundle(path)
    return path


def prepare_inference_artifact(root: Path, directory: Path | None = None) -> Path:
    """Explicit build if absent; validate/reuse a complete artifact without fit."""
    root = root.resolve()
    output = (directory if directory is not None else root / "models/day06").resolve()
    path = output / BUNDLE_NAME
    reference = load_inference_reference(root)
    if output.exists() and any(output.iterdir()):
        bundle = load_bundle(path)
        _check_bundle_reference(bundle, reference)
        _check_build_inputs(reference)
        return path
    return save_bundle(build_inference_bundle(reference), output, reference=reference)


def main() -> None:
    """CLI uses the canonical module, so pickle never stores __main__ classes."""
    import argparse
    from finnews_sentiment.models.inference_artifact import load_bundle as canonical_load
    from finnews_sentiment.models.inference_artifact import prepare_inference_artifact as canonical_prepare

    parser = argparse.ArgumentParser(description="Explicit frozen train-only artifact build or checked reuse")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir
    if output is not None and not output.is_absolute():
        output = args.root.resolve() / output
    path = canonical_prepare(args.root, output)
    bundle = canonical_load(path)
    print(f"Artifact: {path}")
    print(f"Candidate: {bundle.metadata.contract.candidate_id}; train={bundle.metadata.n_train}; features={bundle.metadata.n_features}")
    print(f"Validation replay: {bundle.metadata.validation_replay}; N={bundle.metadata.n_validation}")


if __name__ == "__main__":
    main()
