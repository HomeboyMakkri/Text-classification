"""Read-only frozen inputs and an explicit train-only Day06 build.

Future public operations (implemented in D6-03):
    save_bundle(bundle: InferenceBundle, directory: Path,
                *, reference: InferenceReference) -> Path
    load_bundle(path: Path) -> InferenceBundle

The signatures are design contracts, not executable placeholder APIs.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from finnews_sentiment.data.baseline_config import load_baseline_config
from finnews_sentiment.data.experiment_splits import _object, load_day04_inputs
from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.data.load_processed import _same
from finnews_sentiment.features.labels_duplicates import LABEL_MAPPING
from finnews_sentiment.features import preprocess

BUNDLE_SCHEMA = "day06-inference-v1"


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
