"""Typed raw-text batch inference using a checked, fitted Day06 bundle."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import csv
from io import StringIO
import os
from pathlib import Path
import tempfile

import numpy as np

from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.features.preprocess import preprocess_text
from finnews_sentiment.models.evaluate import _label_ids
from finnews_sentiment.models.inference_artifact import InferenceBundle, _validate_bundle, load_bundle

PREDICTION_COLUMNS = ("text_clean", "pred_label_id", "pred_label")


@dataclass(frozen=True)
class SentimentPrediction:
    """One result in input order; labels describe sentiment, not confidence."""

    text: str
    text_clean: str
    pred_label_id: int
    pred_label: str


def predict_texts(bundle: InferenceBundle, texts: Sequence[str]) -> list[SentimentPrediction]:
    """Validate the entire raw batch, clean with minimal-v1 and predict once.

    A single text must be passed as [text]. Invalid elements reject the whole
    batch before any model calls. Order/duplicates are retained; OOV-only
    nonempty text receives the fitted classifier's ordinary prediction.
    No fit, probability conversion, filtering or string coercion is performed.
    """
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a Sequence[str] batch; pass one text as [text]")
    raw_texts = tuple(texts)
    if not raw_texts:
        raise ValueError("texts batch must not be empty")
    cleaned: list[str] = []
    for index, text in enumerate(raw_texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{index}] must be str, got {type(text).__name__}")
        clean = preprocess_text(text)
        if not clean:
            raise ValueError(f"texts[{index}] is empty after preprocessing")
        cleaned.append(clean)
    _validate_bundle(bundle)
    labels = _label_ids(np.asarray(bundle.pipeline.predict(cleaned)), "inference predictions")
    if len(labels) != len(raw_texts):
        raise ValueError(f"Inference prediction length mismatch: expected {len(raw_texts)}, got {len(labels)}")
    return [
        SentimentPrediction(text, clean, int(label_id), CLASS_ORDER[int(label_id)])
        for text, clean, label_id in zip(raw_texts, cleaned, labels, strict=True)
    ]


def predict_for_file(model_path: Path, input_path: Path, output_path: Path) -> None:
    """Predict an unlabeled UTF-8 CSV, preserving every input field and row.

    Values are read as strings, including NA/nan/numbers. No output exists
    until validation, prediction and serialization succeed. Existing paths
    are never overwritten; input changes during inference abort publication.
    """
    source = input_path.resolve()
    output = output_path.absolute()
    if output.resolve() == source or (output.exists() and source.exists() and output.samefile(source)):
        raise ValueError("Input and output CSV must be different files")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    content = source.read_bytes()
    try:
        reader = csv.reader(StringIO(content.decode("utf-8"), newline=""), strict=True)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV is empty or has no header")
        if any(not column.strip() for column in header):
            raise ValueError("CSV column names must not be blank")
        if len(header) != len(set(header)):
            raise ValueError("CSV column names must be unique")
        if "text" not in header:
            raise ValueError("CSV requires a text column")
        collisions = set(header).intersection(PREDICTION_COLUMNS)
        if collisions:
            raise ValueError(f"CSV already contains prediction columns: {sorted(collisions)}")
        rows: list[list[str]] = []
        for number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"CSV row {number} has {len(row)} fields; expected {len(header)}")
            rows.append(row)
        if not rows:
            raise ValueError("CSV table must contain at least one data row")
    except (UnicodeError, csv.Error, ValueError) as error:
        raise ValueError(f"Invalid input CSV {source}: {error}") from error
    text_index = header.index("text")
    bundle = load_bundle(model_path)
    predictions = predict_texts(bundle, [row[text_index] for row in rows])
    buffer = StringIO(newline="")
    # CRLF also forces quoting of a lone CR inside an original field.
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow([*header, *PREDICTION_COLUMNS])
    for row, prediction in zip(rows, predictions, strict=True):
        writer.writerow([*row, prediction.text_clean, prediction.pred_label_id, prediction.pred_label])
    output_bytes = buffer.getvalue().encode("utf-8")
    if source.read_bytes() != content:
        raise ValueError(f"Input CSV changed during inference: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(dir=output.parent, prefix=".inference-", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(output_bytes)
        # Same-directory hard link publishes complete bytes atomically and
        # fails if another caller created output after the initial check.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an unlabeled UTF-8 CSV using a trusted local fitted bundle")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        predict_for_file(args.model, args.input, args.output)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"Saved predictions: {args.output}")


if __name__ == "__main__":
    main()
