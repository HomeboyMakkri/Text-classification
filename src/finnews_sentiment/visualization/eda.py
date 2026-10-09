from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_subject_distribution(df: pd.DataFrame, subject_col: str = "subject", top_n: int = 10) -> None:
    """Рисует распределение по темам новостей."""
    plt.figure(figsize=(10, 5))
    counts = df[subject_col].value_counts().head(top_n)
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Распределение новостей по subject (top N)")
    plt.tight_layout()
    plt.show()


def plot_provider_distribution(df: pd.DataFrame, provider_col: str = "provider", top_n: int = 10) -> None:
    """Рисует распределение по источникам новостей."""
    plt.figure(figsize=(10, 5))
    counts = df[provider_col].value_counts().head(top_n)
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Распределение новостей по provider (top N)")
    plt.tight_layout()
    plt.show()


def plot_text_length_distribution(df: pd.DataFrame, text_col: str = "text", bins: int = 50) -> None:
    """Распределение длины текстов в словах."""
    lengths = df[text_col].astype(str).str.split().apply(len)
    plt.figure(figsize=(10, 5))
    sns.histplot(lengths, bins=bins, kde=True)
    plt.title("Распределение длины текстов (в словах)")
    plt.xlabel("Количество слов")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.show()