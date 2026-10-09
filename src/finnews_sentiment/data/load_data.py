import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def load_raw_news(path: Path) -> pd.DataFrame:
    """Загрузка сырых данных с новостями из CSV.

    Ожидается как минимум наличие колонок:
    - id (опционально)
    - headline
    - text
    - subject
    - provider
    - tickers (опционально)
    - published_at (опционально)
    - sentiment (если уже размечено)
    """
    df = pd.read_csv(path)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Минимальная очистка данных.

    Здесь можно:
    - убрать полностью пустые строки
    - удалить дубликаты
    - привести названия колонок к единому виду
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates()
    df = df.dropna(subset=["headline", "text"])
    return df


def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(input_path: str, output_path: str, sample: Optional[int] = None) -> None:
    input_p = Path(input_path)
    output_p = Path(output_path)

    df = load_raw_news(input_p)
    if sample is not None and sample > 0:
        df = df.sample(n=sample, random_state=42)
    df = basic_cleaning(df)
    save_processed(df, output_p)
    print(f"Сохранён обработанный датасет: {output_p} (shape={df.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Базовая загрузка и очистка новостей.")
    parser.add_argument("--input", type=str, required=True, help="Путь к исходному CSV с новостями.")
    parser.add_argument("--output", type=str, required=True, help="Путь для сохранения очищенного CSV.")
    parser.add_argument("--sample", type=int, default=None, help="Опционально: взять случайный сэмпл N строк.")
    args = parser.parse_args()

    main(args.input, args.output, sample=args.sample)