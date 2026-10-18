"""Offline Day05 report from saved validation references, annotations and final cache."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import TypedDict

import pandas as pd

from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.features.labels_duplicates import LABEL_MAPPING
from finnews_sentiment.models.confusion import (
    ValidationConfusion, build_validation_confusion, plot_validation_confusion,
)
from finnews_sentiment.models.error_analysis import ValidationAnalysis, load_validation_analysis
from finnews_sentiment.models.failure_cases import (
    FAILURE_CATEGORIES, ValidationExamples, load_failure_annotations, select_validation_examples,
)
from finnews_sentiment.models.final_evaluation import FINAL_NAMES, FinalEvaluation, load_final_evaluation, runtime_versions
from finnews_sentiment.models.run_experiments_day04 import _classification_payload


class ReportSummary(TypedDict):
    """Nested report sections remain typed across notebook cell boundaries."""

    policy: str
    candidate_id: str
    class_order: list[str]
    label_mapping: dict[str, int]
    parameters: dict[str, object]
    fit_policy: str
    metric_policy: object
    preprocessing_policy: object
    validation: dict[str, object]
    manual_review: dict[str, object]
    test: dict[str, object]


class ReportProvenance(TypedDict):
    """Acyclic hashes of inputs and report outputs, excluding provenance itself."""

    policy: str
    class_order: list[str]
    label_mapping: dict[str, int]
    runtime_versions: dict[str, str]
    input_checksums: dict[str, str]
    output_checksums: dict[str, str]
    frozen_protocol_sha256: str
    final_evaluation_sha256: str
    policies: dict[str, str]


@dataclass(frozen=True)
class Day05Report:
    """Reviewable report contract shared by CLI, tests and notebook cells."""

    analysis: ValidationAnalysis
    confusion: ValidationConfusion
    selection: ValidationExamples
    reviewed_examples: pd.DataFrame
    final: FinalEvaluation
    summary: ReportSummary
    provenance: ReportProvenance
    artifacts: dict[str, Path]


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _snapshot(root: Path, final: FinalEvaluation) -> dict[str, bytes]:
    recorded = final.protocol.get("input_checksums")
    if not isinstance(recorded, dict) or any(not isinstance(name, str) for name in recorded):
        raise ValueError("Frozen protocol must record input paths")
    names = sorted(recorded)
    names.extend(f"reports/day05/{name}" for name in FINAL_NAMES)
    names.append("docs/day05_failure_cases.csv")
    contents = {name: (root / name).read_bytes() for name in names}
    for name, expected in recorded.items():
        if _sha(contents[name]) != expected:
            raise ValueError(f"Report input checksum differs from frozen protocol: {root / name}")
    return contents


def _unchanged(root: Path, contents: dict[str, bytes]) -> None:
    for name, expected in contents.items():
        if (root / name).read_bytes() != expected:
            raise ValueError(f"Report input changed during build/write: {root / name}")


def _csv_bytes(frame: pd.DataFrame, *, index: bool = False) -> bytes:
    """Read back with declared dtypes, preserving empty texts and booleans."""
    content = frame.to_csv(index=index, lineterminator="\n").encode("utf-8")
    dtypes = {str(column): dtype for column, dtype in frame.dtypes.items()}
    if index:
        dtypes[str(frame.index.name)] = frame.index.dtype
    restored = pd.read_csv(
        BytesIO(content), keep_default_na=False, dtype=dtypes, float_precision="round_trip",
        index_col=str(frame.index.name) if index else None,
    )
    if index:
        restored.columns.name = frame.columns.name
    try:
        pd.testing.assert_frame_equal(restored, frame, check_exact=True)
    except AssertionError as error:
        raise ValueError("Day05 report CSV roundtrip mismatch") from error
    return content


def _summary(
    analysis: ValidationAnalysis, confusion: ValidationConfusion,
    selection: ValidationExamples, reviewed: pd.DataFrame, final: FinalEvaluation,
) -> ReportSummary:
    n_reviewed = len(reviewed)
    categories = [
        {"category": category, "count": int(reviewed["primary_category"].eq(category).sum()),
         "reviewed_rows": n_reviewed,
         "fraction_of_reviewed": float(reviewed["primary_category"].eq(category).sum() / n_reviewed) if n_reviewed else 0.0}
        for category in FAILURE_CATEGORIES
    ]
    return {
        "policy": "day05-analysis-summary-v1", "candidate_id": analysis.candidate_id,
        "class_order": list(CLASS_ORDER), "label_mapping": LABEL_MAPPING,
        "parameters": analysis.parameters, "fit_policy": "day03_train_only",
        "metric_policy": final.protocol["metric_policy"],
        "preprocessing_policy": final.protocol["preprocessing_policy"],
        "validation": {
            "evaluation_split": "validation", **_classification_payload(confusion.metrics),
            "n_errors": len(analysis.errors), "n_correct": len(analysis.table) - len(analysis.errors),
            "error_fraction_of_validation": len(analysis.errors) / len(analysis.table),
            "n_groups": int(analysis.table["group_id"].nunique()),
            "n_error_groups": int(analysis.errors["group_id"].nunique()),
            "retained_duplicate_rows": len(analysis.table) - int(analysis.table["group_id"].nunique()),
            "counts": confusion.counts.to_numpy().tolist(),
            "normalized_by_true_support": confusion.normalized.to_numpy().tolist(),
            "confusion_pairs": confusion.pairs.to_dict("records"),
        },
        "manual_review": {
            "selection_policy": "first three source_row_id per directed pair; fixed class order",
            "n_reviewed_rows": n_reviewed, "n_reviewed_groups": int(reviewed["group_id"].nunique()),
            "repeated_group_rows_in_examples": n_reviewed - int(reviewed["group_id"].nunique()),
            "selected_rows_from_duplicate_validation_groups": int(reviewed["is_duplicate_group"].sum()),
            "source_row_ids": reviewed["source_row_id"].tolist(),
            "pair_selection": selection.pairs.to_dict("records"), "category_counts": categories,
            "category_denominator": "reviewed rows only; not all errors or all validation rows",
        },
        "test": {
            "evaluation_split": "test", "cache_status": final.cache_status,
            **_classification_payload(final.confusion.metrics),
            "counts": final.confusion.counts.to_numpy().tolist(),
            "normalized_by_true_support": final.confusion.normalized.to_numpy().tolist(),
            "n_train": final.n_train, "n_features": final.n_features,
        },
    }


def _observations(
    analysis: ValidationAnalysis, confusion: ValidationConfusion,
    selection: ValidationExamples, reviewed: pd.DataFrame, final: FinalEvaluation,
) -> str:
    lines = [
        "# Day05: validation analysis and frozen final test", "",
        f"Candidate: `{analysis.candidate_id}`; fit policy: `day03_train_only`.",
        "Sources: Day02 processed/provenance, Day03 manifest/reference, Day04 selection/config/reference,",
        "`docs/day05_failure_cases.csv` and verified `reports/day05/final_evaluation.json`.", "",
        "## Validation (saved predictions)", "",
        f"N={len(analysis.table)}; correct={len(analysis.table) - len(analysis.errors)}; errors={len(analysis.errors)}; "
        f"accuracy={confusion.metrics.accuracy:.6f}; macro F1={confusion.metrics.macro_f1:.6f}.",
        f"Groups: validation={analysis.table['group_id'].nunique()}, errors={analysis.errors['group_id'].nunique()}.",
        "Rows=true, columns=predicted; class order: negative, neutral, positive.",
        "Directed fractions use true class support, not all errors:", "",
    ]
    for pair in confusion.pairs.to_dict("records"):
        lines.append(f"- {pair['true_label']} → {pair['pred_label']}: {pair['count']}/{pair['true_support']} "
                     f"(fraction_of_true={pair['fraction_of_true']:.6f}).")
    lines.extend(["", "### Manual review: selected validation examples", "",
                  f"Reviewed {len(reviewed)} rows / {reviewed['group_id'].nunique()} groups; "
                  f"duplicate-group rows in examples={len(reviewed) - reviewed['group_id'].nunique()}.",
                  "Category denominators are reviewed rows only. No causal explanation is established.", ""])
    for category in FAILURE_CATEGORIES:
        count = int(reviewed["primary_category"].eq(category).sum())
        if count:
            lines.append(f"- {category}: {count}/{len(reviewed)} reviewed rows.")
    for pair in selection.pairs.to_dict("records"):
        if pair["status"] == "empty":
            lines.append(f"- Empty pair: {pair['true_label']} → {pair['pred_label']} (0 examples).")
    for row in reviewed.to_dict("records"):
        lines.extend(["", f"#### source_row_id={row['source_row_id']}; group_id={row['group_id']}; "
                      f"{row['true_label']} → {row['pred_label']}", "",
                      f"Primary category: {row['primary_category']}.",
                      f"Observation: {row['observation']}",
                      f"Hypothesis (not established): {row['hypothesis'] or 'not specified'}",
                      f"Rationale: {row['rationale']}"])
    lines.extend(["", "Future hypotheses require separate predeclared train-only group-aware CV experiments;",
                  "learned transformations must fit inside each fold Pipeline. Current model/labels stay frozen.",
                  "Reviewed validation is not an independent new confirmation set; see tracked `docs/day05.md`.",
                  "", "## Frozen held-out test (cache only)", "",
                  f"Cache status: {final.cache_status}; N_test={len(final.predictions)}; "
                  f"accuracy={final.confusion.metrics.accuracy:.6f}; macro F1={final.confusion.metrics.macro_f1:.6f}.",
                  "No fit/predict/CV/download or new test evaluation was performed by this report.", ""])
    for item in final.confusion.metrics.per_class:
        lines.append(f"- {item.class_name}: precision={item.precision:.6f}, recall={item.recall:.6f}, "
                     f"F1={item.f1:.6f}, support={item.support}.")
    lines.extend(["", "## Limits", "",
                  "Validation and test contain different rows; they are not a paired experiment.",
                  "Category counts describe the selected examples, not all errors; duplicate rows are not independent observations.",
                  "One split and a small holdout limit generalization; CV std is not a confidence interval.",
                  "Test diagnostics are descriptive and do not select parameters, thresholds, seed or preprocessing.",
                  "Decision scores are not probabilities; live financial quality and metric gains are not promised.",
                  "The final completion marker is not a multi-file transaction or an exactly-once guarantee after a crash.", ""])
    return "\n".join(lines)


def run_error_analysis_day05(root: Path) -> Day05Report:
    """Build only Day05 report outputs; missing/bad final cache never triggers fit."""
    root = root.resolve()
    final = load_final_evaluation(root)
    contents = _snapshot(root, final)
    analysis = load_validation_analysis(root)
    confusion = build_validation_confusion(analysis.table)
    selection = select_validation_examples(analysis.table)
    reviewed = load_failure_annotations(root / "docs/day05_failure_cases.csv", selection.examples)
    if analysis.candidate_id != final.candidate_id:
        raise ValueError("Validation and final cache candidates differ")
    if (int(confusion.counts.to_numpy().sum()) != len(analysis.table)
            or int(confusion.pairs["count"].sum()) != len(analysis.errors)
            or not selection.pairs["available_errors"].equals(confusion.pairs["count"])):
        raise ValueError("Validation confusion/errors/example denominators are inconsistent")
    summary = _summary(analysis, confusion, selection, reviewed, final)
    payloads = {
        "validation_analysis.csv": _csv_bytes(analysis.table),
        "validation_errors.csv": _csv_bytes(analysis.errors.reset_index(drop=True)),
        "validation_examples.csv": _csv_bytes(reviewed),
        "confusion_pairs.csv": _csv_bytes(confusion.pairs),
        "analysis_summary.json": _json_bytes(summary),
        "observations.md": _observations(analysis, confusion, selection, reviewed, final).encode("utf-8"),
    }
    if json.loads(payloads["analysis_summary.json"]) != summary:
        raise ValueError("Day05 summary JSON roundtrip mismatch")
    for kind, frame in (("counts", confusion.counts), ("normalized", confusion.normalized)):
        payloads[f"validation_confusion_{kind}.csv"] = _csv_bytes(frame, index=True)
        figure = plot_validation_confusion(confusion, normalized=kind == "normalized")
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=140)
        figure.clear()
        payloads[f"validation_confusion_{kind}.png"] = buffer.getvalue()
    directory = root / "reports/day05"
    artifacts = {name: directory / name for name in payloads}
    if set(contents).intersection(str(path.relative_to(root)) for path in artifacts.values()):
        raise ValueError("Day05 report outputs must not overwrite inputs/final cache")
    input_hashes = {name: _sha(content) for name, content in contents.items()}
    provenance: ReportProvenance = {
        "policy": "day05-report-provenance-v1", "class_order": list(CLASS_ORDER),
        "label_mapping": LABEL_MAPPING, "runtime_versions": runtime_versions(),
        "input_checksums": input_hashes,
        "output_checksums": {name: _sha(content) for name, content in payloads.items()},
        "frozen_protocol_sha256": input_hashes["reports/day05/frozen_protocol.json"],
        "final_evaluation_sha256": input_hashes["reports/day05/final_evaluation.json"],
        "policies": {"fit": "none (report-only)", "test": "verified cache only",
                     "selection": "first three source IDs per directed validation pair",
                     "row_metrics": "retained rows including duplicates", "category_denominator": "reviewed rows only"},
    }
    provenance_bytes = _json_bytes(provenance)
    if json.loads(provenance_bytes) != provenance:
        raise ValueError("Day05 provenance JSON roundtrip mismatch")
    _unchanged(root, contents)
    # All checks precede writes; sequential output writes are not a transaction.
    for name, content in payloads.items():
        artifacts[name].write_bytes(content)
    for name, content in payloads.items():
        if artifacts[name].read_bytes() != content:
            raise ValueError(f"Day05 output read-back mismatch: {artifacts[name]}")
    _unchanged(root, contents)
    provenance_path = directory / "provenance.json"
    provenance_path.write_bytes(provenance_bytes)
    if provenance_path.read_bytes() != provenance_bytes:
        raise ValueError(f"Day05 provenance read-back mismatch: {provenance_path}")
    artifacts["provenance.json"] = provenance_path
    return Day05Report(analysis, confusion, selection, reviewed, final, summary, provenance, artifacts)


def main() -> None:
    """Report-only CLI: never dispatch the explicit final-evaluation command."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    result = run_error_analysis_day05(args.root)
    print(f"Candidate: {result.analysis.candidate_id}; final cache: {result.final.cache_status}")
    print(f"Validation: {len(result.analysis.table)} rows, {len(result.analysis.errors)} errors; "
          f"reviewed: {len(result.reviewed_examples)} rows / {result.reviewed_examples['group_id'].nunique()} groups")
    print(f"Cached test: {len(result.final.predictions)} rows; macro F1={result.final.confusion.metrics.macro_f1:.6f}")
    print(f"Report summary: {result.artifacts['analysis_summary.json']}")


if __name__ == "__main__":
    main()
