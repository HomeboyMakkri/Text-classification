"""Offline contract checks; these fixtures are not the learning dataset."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd
from pandas.testing import assert_frame_equal

from finnews_sentiment.data.load_data import load_raw_news, validate_news
from finnews_sentiment.data.prepare_phrasebank import parse_phrasebank, prepare_phrasebank
from finnews_sentiment.visualization.eda import (
    class_distribution,
    class_examples,
    plot_class_distribution,
    plot_text_length_distribution,
    quality_summary,
    text_lengths,
)


class Day01Tests(unittest.TestCase):
    def test_csv_preserves_raw_rows_and_literal_na(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text('text,sentiment\nNA,neutral\n,negative\n"   ",positive\nNA,neutral\n', encoding="utf-8")
            df = load_raw_news(path)
        self.assertEqual(len(df), 4)
        self.assertEqual(df.loc[0, "text"], "NA")
        self.assertTrue(pd.isna(df.loc[1, "text"]))
        self.assertEqual(df.loc[2, "text"], "   ")
        self.assertEqual(quality_summary(df)["duplicate_rows"], 1)

    def test_missing_file_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_raw_news(Path(directory) / "absent.csv")

    def test_invalid_schema_labels_and_types(self) -> None:
        cases = [
            pd.DataFrame({"headline": ["news"], "sentiment": ["neutral"]}),
            pd.DataFrame({"text": [], "sentiment": []}),
            pd.DataFrame({"text": ["news"], "sentiment": ["bullish"]}),
            pd.DataFrame({"text": [123], "sentiment": ["neutral"]}),
            pd.DataFrame({"text": ["news"], "sentiment": [1]}),
        ]
        for df in cases:
            with self.subTest(df=df):
                with self.assertRaises(ValueError):
                    validate_news(df)

    def test_missing_blank_duplicates_and_conflicts(self) -> None:
        df = pd.DataFrame({
            "text": ["same", "same", "same", None, "  ", "other"],
            "sentiment": ["negative", "negative", "positive", "neutral", None, "neutral"],
        })
        before = df.copy(deep=True)
        report = quality_summary(df)
        self.assertEqual(report["missing"], {"text": 1, "sentiment": 1})
        self.assertEqual(report["blank_texts"], 1)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["duplicate_texts"], 2)
        self.assertEqual(report["conflicting_texts"], 1)
        assert_frame_equal(df, before)

    def test_absent_class_and_missing_labels_have_clear_denominator(self) -> None:
        df = pd.DataFrame({"text": ["a", "b"], "sentiment": ["neutral", None]})
        distribution = class_distribution(df)
        self.assertEqual(distribution.index.tolist(), ["negative", "neutral", "positive"])
        self.assertEqual(distribution["count"].tolist(), [0, 1, 0])
        self.assertEqual(distribution["fraction"].sum(), 0.5)
        self.assertEqual(distribution["count"].sum() + df["sentiment"].isna().sum(), len(df))

    def test_lengths_are_raw_and_keep_missing(self) -> None:
        df = pd.DataFrame({"text": ["A  B!", "  ", None], "sentiment": ["negative"] * 3})
        lengths = text_lengths(df)
        self.assertEqual(lengths.iloc[0].tolist(), [5, 2])
        self.assertEqual(lengths.iloc[1].tolist(), [2, 0])
        self.assertTrue(lengths.loc[2].isna().all())

    def test_all_texts_missing_still_has_a_quality_report(self) -> None:
        df = pd.DataFrame({"text": [None], "sentiment": [None]})
        self.assertEqual(quality_summary(df)["blank_texts"], 0)
        self.assertTrue(text_lengths(df).isna().all().all())
        self.assertEqual(class_examples(df), {"negative": [], "neutral": [], "positive": []})

    def test_examples_are_deterministic_and_skip_blanks(self) -> None:
        df = pd.DataFrame({"text": ["  ", "first", "second"], "sentiment": ["neutral"] * 3})
        self.assertEqual(class_examples(df, n=1)["neutral"], ["first"])
        with self.assertRaises(ValueError):
            class_examples(df, n=0)

    def test_source_parser_preserves_internal_at_sign_and_spaces(self) -> None:
        df = parse_phrasebank("Contact a@b.com  @neutral\nProfit rose@positive\n")
        self.assertEqual(df.loc[0, "text"], "Contact a@b.com  ")
        self.assertEqual(df["sentiment"].tolist(), ["neutral", "positive"])

    def test_source_parser_rejects_bad_records(self) -> None:
        for content in ("", "no separator", " @neutral", "text@unknown"):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    parse_phrasebank(content)

    def test_cached_archive_conversion_roundtrip_and_tamper_detection(self) -> None:
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("sample.txt", "Caf\xe9 grows@positive\nSales fall@negative\n".encode("iso-8859-1"))
        payload = buffer.getvalue()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "data/raw").mkdir(parents=True)
            archive_path = root / "data/raw/FinancialPhraseBank-v1.0.zip"
            archive_path.write_bytes(payload)
            config = {
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
                "member": "sample.txt", "encoding": "iso-8859-1", "expected_rows": 2,
            }
            (root / "configs/dataset.json").write_text(json.dumps(config))
            first = load_raw_news(prepare_phrasebank(root))
            second = load_raw_news(prepare_phrasebank(root))
            assert_frame_equal(first, second)
            self.assertEqual(first.loc[0, "text"], "Caf\xe9 grows")
            config["expected_rows"] = 3
            (root / "configs/dataset.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "row count"):
                prepare_phrasebank(root)
            archive_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                prepare_phrasebank(root)
            archive_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "download"):
                prepare_phrasebank(root)

    def test_plots_write_png_and_leave_input_unchanged(self) -> None:
        df = pd.DataFrame({"text": ["Profit rose", "Sales fell"], "sentiment": ["positive", "negative"]})
        before = df.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            for function, name in ((plot_class_distribution, "classes"), (plot_text_length_distribution, "lengths")):
                path = Path(directory) / "nested" / f"{name}.png"
                fig = function(df, path)
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))
                plt.close(fig)
        assert_frame_equal(df, before)


if __name__ == "__main__":
    unittest.main()
