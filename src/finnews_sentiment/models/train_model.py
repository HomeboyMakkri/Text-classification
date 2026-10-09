import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from finnews_sentiment.features.preprocess import clean_news_data, add_basic_features


def load_processed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_baseline_pipeline() -> Pipeline:
    """Создаёт простой пайплайн TF-IDF + Logistic Regression."""
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    pipe = Pipeline(
        [
            ("tfidf", tfidf),
            ("clf", clf),
        ]
    )
    return pipe


def train_baseline(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    target_col: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, float]:
    df = df.dropna(subset=[text_col, target_col])
    X = df[text_col].astype(str)
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_baseline_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_valid)
    f1 = f1_score(y_valid, y_pred, average="macro")

    print("Classification report (validation):")
    print(classification_report(y_valid, y_pred))
    print(f"Macro F1: {f1:.4f}")

    return pipe, f1


def main(input_path: str, model_path: str) -> None:
    input_p = Path(input_path)
    model_p = Path(model_path)
    model_p.parent.mkdir(parents=True, exist_ok=True)

    df = load_processed(input_p)
    df = clean_news_data(df, text_col="text")
    df = add_basic_features(df)

    model, f1 = train_baseline(df)

    meta = {"macro_f1": float(f1)}

    joblib.dump({"model": model, "meta": meta}, model_p)
    print(f"Сохранена модель: {model_p} (macro F1={f1:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение базовой модели тональности новостей.")
    parser.add_argument("--input", type=str, required=True, help="Путь к очищенному CSV.")
    parser.add_argument("--model", type=str, required=True, help="Путь для сохранения модели (joblib).")
    args = parser.parse_args()

    main(args.input, args.model)