"""Frozen Day06 inputs; no fitting, prediction or artifact writes.

Future public operations (implemented in D6-02/D6-03):
    build_inference_bundle(reference: InferenceReference) -> InferenceBundle
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

import pandas as pd
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
    """Future successful Day06 build evidence, distinct from cached Day05 replay."""

    contract: FrozenInferenceContract
    n_features: int
    validation_replay: Literal["exact_match"]


@dataclass(frozen=True)
class InferenceBundle:
    """Future fitted TF-IDF/classifier Pipeline and its checked metadata."""

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
