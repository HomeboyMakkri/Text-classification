"""Fixed-order confusion summaries for already aligned validation predictions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.models.evaluate import ClassificationMetrics, evaluate_predictions


@dataclass(frozen=True)
class ValidationConfusion:
    """True rows, predicted columns, and six directed error pairs."""

    counts: pd.DataFrame
    normalized: pd.DataFrame
    pairs: pd.DataFrame
    per_class: pd.DataFrame
    metrics: ClassificationMetrics


def build_validation_confusion(table: pd.DataFrame) -> ValidationConfusion:
    """Summarize saved labels; keep absent classes and zero-support rows explicit.

    Fractions use true support, including correct predictions, as denominator.
    Each retained row contributes once, including retained duplicate text groups.
    """
    required = ("true_label_id", "pred_label_id")
    if any(column not in table for column in required):
        raise ValueError(f"Validation analysis must contain {required}")
    truth = table["true_label_id"].to_numpy()
    predicted = table["pred_label_id"].to_numpy()
    metrics = evaluate_predictions(truth, predicted)
    counts = confusion_matrix(truth, predicted, labels=[0, 1, 2])
    support = counts.sum(axis=1)
    normalized = np.divide(
        counts, support[:, None], out=np.zeros((3, 3), dtype=float), where=support[:, None] != 0,
    )
    rows = pd.Index(CLASS_ORDER, name="true_label")
    columns = pd.Index(CLASS_ORDER, name="pred_label")
    pairs = pd.DataFrame([
        {
            "true_label": true_label, "pred_label": pred_label,
            "count": int(counts[i, j]), "true_support": int(support[i]),
            "fraction_of_true": float(normalized[i, j]),
        }
        for i, true_label in enumerate(CLASS_ORDER)
        for j, pred_label in enumerate(CLASS_ORDER) if i != j
    ])
    per_class = pd.DataFrame([
        {"class_id": item.class_id, "class_name": item.class_name,
         "precision": item.precision, "recall": item.recall, "f1": item.f1,
         "support": item.support, "predicted_count": int(counts[:, item.class_id].sum())}
        for item in metrics.per_class
    ])
    return ValidationConfusion(
        pd.DataFrame(counts, index=rows, columns=columns),
        pd.DataFrame(normalized, index=rows, columns=columns), pairs, per_class, metrics,
    )


def plot_validation_confusion(result: ValidationConfusion, *, normalized: bool = False) -> Figure:
    """Create an Agg figure with denominators, orientation and absent-class labels."""
    values = result.normalized if normalized else result.counts
    numeric_values = values.to_numpy(dtype=float)
    support = result.counts.sum(axis=1).to_numpy()
    figure = Figure(figsize=(8, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.imshow(numeric_values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    figure.colorbar(image, ax=axis, label="Fraction of true class" if normalized else "Rows")
    axis.set(
        xticks=range(3), xticklabels=list(CLASS_ORDER), yticks=range(3),
        yticklabels=[
            f"{label} (support={int(n)}{' — absent' if n == 0 else ''})"
            for label, n in zip(CLASS_ORDER, support, strict=True)
        ],
        xlabel="Predicted label", ylabel="True label",
        title=f"Validation confusion — {'row-normalized (count / true support)' if normalized else 'counts'}\nN={result.metrics.n_samples}",
    )
    maximum = float(numeric_values.max())
    for i in range(3):
        for j in range(3):
            value = float(numeric_values[i, j])
            text = f"{value:.1%}" if normalized else str(int(value))
            axis.text(j, i, text, ha="center", va="center", color="white" if value > maximum / 2 else "black")
    return figure


def save_validation_confusion(result: ValidationConfusion, output_dir: Path) -> dict[str, Path]:
    """Explicitly save Day05 CSV/PNG outputs without GUI or model execution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for kind, frame in (("counts", result.counts), ("normalized", result.normalized)):
        stem = f"validation_confusion_{kind}"
        csv_path = output_dir / f"{stem}.csv"
        frame.to_csv(csv_path, index=True, lineterminator="\n")
        paths[f"{kind}_csv"] = csv_path
        png_path = output_dir / f"{stem}.png"
        figure = plot_validation_confusion(result, normalized=kind == "normalized")
        figure.savefig(png_path, dpi=140)
        figure.clear()
        paths[f"{kind}_png"] = png_path
    pairs_path = output_dir / "confusion_pairs.csv"
    result.pairs.to_csv(pairs_path, index=False, lineterminator="\n")
    paths["pairs_csv"] = pairs_path
    return paths
