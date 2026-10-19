"""Typed raw-text batch inference using a checked, fitted Day06 bundle."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.features.preprocess import preprocess_text
from finnews_sentiment.models.evaluate import _label_ids
from finnews_sentiment.models.inference_artifact import InferenceBundle, _validate_bundle


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
    """Explicit boundary: CSV adapter/CLI is scheduled for D6-06."""
    raise NotImplementedError("CSV inference is scheduled for D6-06; use load_bundle + predict_texts for raw-text batches")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV inference adapter (scheduled for D6-06)")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        predict_for_file(args.model, args.input, args.output)
    except NotImplementedError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
