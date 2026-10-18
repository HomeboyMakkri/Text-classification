"""Deterministic validation examples and checked, manually authored annotations."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from finnews_sentiment.data.load_data import CLASS_ORDER

ANALYSIS_COLUMNS = (
    "source_row_id", "group_id", "text", "text_clean", "true_label_id",
    "pred_label_id", "true_label", "pred_label", "is_error",
)
ANNOTATION_COLUMNS = (
    "source_row_id", "primary_category", "observation", "hypothesis", "rationale",
)
FAILURE_CATEGORIES = (
    "mixed sentiment", "financial context", "negation/comparison",
    "insufficient context/ambiguous label", "preprocessing/representation limitation",
    "other/unclear",
)


@dataclass(frozen=True)
class ValidationExamples:
    """At most three rows per directed pair, with full-validation group counts."""

    examples: pd.DataFrame
    pairs: pd.DataFrame


def select_validation_examples(table: pd.DataFrame) -> ValidationExamples:
    """Select first three source IDs per error pair, never deduplicating text.

    Input is the verified D5-01 validation table. An empty table or no errors
    yields empty examples and six explicit pair rows. Group counts distinguish
    repeated rows in validation, all errors, and the selected examples.
    """
    if any(column not in table for column in ANALYSIS_COLUMNS):
        raise ValueError(f"Validation analysis must contain {ANALYSIS_COLUMNS}")
    if table[list(ANALYSIS_COLUMNS)].isna().any().any():
        raise ValueError("Validation analysis contains missing values")
    for column in ("source_row_id", "group_id", "true_label_id", "pred_label_id"):
        if not pd.api.types.is_integer_dtype(table[column].dtype) or pd.api.types.is_bool_dtype(table[column].dtype):
            raise ValueError(f"Validation {column} must contain integer IDs")
        if table[column].lt(0).any():
            raise ValueError(f"Validation {column} must contain nonnegative IDs")
    if table["source_row_id"].duplicated().any():
        raise ValueError("Validation source_row_id must be unique")
    for column in ("text", "text_clean"):
        if not table[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Validation {column} must contain nonempty text")
    mapping = dict(enumerate(CLASS_ORDER))
    for prefix in ("true", "pred"):
        if not table[f"{prefix}_label_id"].isin((0, 1, 2)).all():
            raise ValueError(f"Unknown {prefix} class IDs")
        if not table[f"{prefix}_label"].eq(table[f"{prefix}_label_id"].map(mapping)).all():
            raise ValueError(f"Inconsistent {prefix} label names")
    if not pd.api.types.is_bool_dtype(table["is_error"].dtype) or not table["is_error"].eq(
        table["true_label_id"].ne(table["pred_label_id"])
    ).all():
        raise ValueError("Validation is_error must match true/pred labels")
    errors = table.loc[table["is_error"]].sort_values("source_row_id", kind="stable")
    selected: list[pd.DataFrame] = []
    pair_rows: list[dict[str, object]] = []
    for true_label in CLASS_ORDER:
        for pred_label in CLASS_ORDER:
            if true_label == pred_label:
                continue
            pair = errors.loc[errors["true_label"].eq(true_label) & errors["pred_label"].eq(pred_label)]
            examples = pair.head(3)
            selected.append(examples)
            pair_rows.append({
                "true_label": true_label, "pred_label": pred_label,
                "available_errors": len(pair), "selected_examples": len(examples),
                "selected_groups": int(examples["group_id"].nunique()),
                "status": "empty" if pair.empty else "selected",
            })
    examples = pd.concat(selected, ignore_index=True).loc[:, list(ANALYSIS_COLUMNS)].copy()
    for name, frame in (("validation", table), ("error", errors), ("selected", examples.copy())):
        counts = frame["group_id"].value_counts()
        examples[f"group_{name}_rows"] = examples["group_id"].map(counts).astype("int64")
    examples["is_duplicate_group"] = examples["group_validation_rows"].gt(1)
    return ValidationExamples(examples, pd.DataFrame(pair_rows))


def validate_failure_annotations(examples: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Require exactly one annotation per selected validation ID, joined by ID.

    Extra IDs (including train/test IDs), missing or duplicate IDs are rejected.
    Annotation file order is immaterial; output preserves example pair/ID order.
    """
    if "source_row_id" not in examples or examples["source_row_id"].duplicated().any():
        raise ValueError("Examples must have unique source_row_id values")
    if tuple(annotations.columns) != ANNOTATION_COLUMNS:
        raise ValueError(f"Annotation columns must be {ANNOTATION_COLUMNS}")
    ids = annotations["source_row_id"].astype("string")
    if not ids.str.fullmatch(r"[0-9]+", na=False).all():
        raise ValueError("Annotation source_row_id must contain nonnegative integer literals")
    checked = annotations.copy(deep=True)
    checked["source_row_id"] = ids.astype("int64")
    if checked["source_row_id"].duplicated().any():
        raise ValueError("Duplicate annotation source_row_id")
    expected = set(examples["source_row_id"])
    actual = set(checked["source_row_id"])
    if actual != expected:
        raise ValueError(f"Annotation IDs mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    if not checked["primary_category"].isin(FAILURE_CATEGORIES).all():
        raise ValueError(f"Annotation primary_category must be one of {FAILURE_CATEGORIES}")
    for column in ("observation", "rationale"):
        if not checked[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Annotation {column} must be nonempty text")
    if not checked["hypothesis"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Annotation hypothesis must be text; empty string is allowed")
    return examples.merge(checked, on="source_row_id", how="left", validate="one_to_one", sort=False)


def load_failure_annotations(path: Path, examples: pd.DataFrame) -> pd.DataFrame:
    """Read reviewed UTF-8 annotations without treating an empty hypothesis as NaN."""
    try:
        annotations = pd.read_csv(path, dtype="str", keep_default_na=False, encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing manual annotations {path}; review the selected validation examples and create {ANNOTATION_COLUMNS}") from error
    try:
        return validate_failure_annotations(examples, annotations)
    except ValueError as error:
        raise ValueError(f"Invalid manual annotations {path}: {error}") from error
