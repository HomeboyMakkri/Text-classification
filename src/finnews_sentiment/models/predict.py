import argparse
from pathlib import Path
from typing import List

import joblib
import pandas as pd

from finnews_sentiment.features.preprocess import clean_news_data, add_basic_features


def load_model(path: Path):
    bundle = joblib.load(path)
    return bundle["model"], bundle.get("meta", {})


def predict_for_file(model_path: str, input_path: str, output_path: str) -> None:
    model_p = Path(model_path)
    input_p = Path(input_path)
    output_p = Path(output_path)

    model, meta = load_model(model_p)
    df = pd.read_csv(input_p)

    df = clean_news_data(df, text_col="text")
    df = add_basic_features(df)

    preds = model.predict(df["text_clean"].astype(str))
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df["text_clean"].astype(str))

    df_out = df.copy()
    df_out["pred_sentiment"] = preds
    if proba is not None:
        for i, cls in enumerate(model.classes_):
            df_out[f"proba_{cls}"] = proba[:, i]

    output_p.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_p, index=False)
    print(f"Сохранён файл с предсказаниями: {output_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс модели для файла с новостями.")
    parser.add_argument("--model", type=str, required=True, help="Путь к модели (joblib).")
    parser.add_argument("--input", type=str, required=True, help="Путь к CSV с новыми новостями.")
    parser.add_argument("--output", type=str, required=True, help="Путь для сохранения CSV с предсказаниями.")
    args = parser.parse_args()

    predict_for_file(args.model, args.input, args.output)