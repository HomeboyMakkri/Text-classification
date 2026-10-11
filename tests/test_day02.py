"""Offline minimal-v1 boundaries; fixtures are not notebook data."""

from pathlib import Path
import subprocess
import sys
from typing import cast
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.features.preprocess import (
    EmptyProcessedDatasetError, clean_news_data, preprocess_news, preprocess_text,
)


class PreprocessTextTests(unittest.TestCase):
    def test_preserves_financial_meaning_and_negation(self) -> None:
        raw = "  Profit is NOT down -5% &amp; costs £2.50, €3 or $4. AAPL isn't up; no, never.  "
        expected = "profit is not down -5% & costs £2.50, €3 or $4. aapl isn't up; no, never."
        self.assertEqual(preprocess_text(raw), expected)

    def test_unicode_whitespace_and_case(self) -> None:
        self.assertEqual(preprocess_text("\tEUR\n  13.1\r\nmn\u00a0+2%\u2003Sales "), "eur 13.1 mn +2% sales")

    def test_tags_separate_words_and_keep_element_content(self) -> None:
        cases = {
            "<b>No</b> growth": "no growth",
            'Profit<a title="x > y" href="/news">rose</a>today': "profit rose today",
            "A<!-- comment\nignored -->B<br/>C": "a b c",
            "<div data-x='a > b' disabled>USD</div>": "usd",
            "<script>NOT up</script>": "not up",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(preprocess_text(raw), expected)

    def test_comparisons_and_incomplete_markup_are_preserved(self) -> None:
        for raw in ("Profit < 5 and growth > 3", "Profit <b", '<b title="unfinished', "Profit <!-- unfinished"):
            with self.subTest(raw=raw):
                self.assertEqual(preprocess_text(raw), raw.lower())

    def test_entities_nested_numeric_unknown_and_case(self) -> None:
        cases = {
            "A&nbsp;B &#37; &#x20AC;": "a b % €",
            "&amp;lt;b&amp;gt;NO&amp;lt;/b&amp;gt;": "no",
            "&AmP; costs": "& costs",
            "&notareal; &madeup;": "&notareal; &madeup;",
            "A &amp B": "a & b",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(preprocess_text(raw), expected)

    def test_empty_inputs_and_markup_only_return_empty(self) -> None:
        for raw in ("", " \t\n\u00a0", "<b></b>", "<!-- nothing -->", "&nbsp;"):
            with self.subTest(raw=raw):
                self.assertEqual(preprocess_text(raw), "")

    def test_non_strings_raise_instead_of_silent_coercion(self) -> None:
        # Runtime boundary deliberately bypasses the annotated str input.
        for value in (None, 1, float("nan"), ["news"], b"news"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "text must be str"):
                    preprocess_text(cast(str, value))

    def test_idempotent_and_repeatable(self) -> None:
        cases = (
            "Profit is NOT down -5% &amp; costs £2.50.",
            "&amp;amp;lt;b&amp;amp;gt;NO&amp;amp;lt;/b&amp;amp;gt;",
            "&AmP; &notareal; <b title='>'>EUR</b>\u00a0 2.5",
            "A < 5 and B > 3", "", "\tGrowth\n+10%",
        )
        for raw in cases:
            with self.subTest(raw=raw):
                cleaned = preprocess_text(raw)
                self.assertEqual(preprocess_text(cleaned), cleaned)
                self.assertEqual(preprocess_text(raw), cleaned)

    def test_import_and_call_without_nltk_or_network(self) -> None:
        code = '''
import importlib.abc
import socket
import sys
from unittest.mock import patch

class NoNltk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "nltk" or fullname.startswith("nltk."):
            raise AssertionError("NLTK must not be imported")

sys.meta_path.insert(0, NoNltk())
sys.path.insert(0, sys.argv[1])
with patch.object(socket, "socket", side_effect=AssertionError("Network forbidden")):
    from finnews_sentiment.features.preprocess import preprocess_text
    assert preprocess_text("NOT -5%") == "not -5%"
assert "nltk" not in sys.modules
'''
        result = subprocess.run(
            [sys.executable, "-c", code, str(ROOT / "src")],
            capture_output=True, text=True, check=False, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class PreprocessNewsTests(unittest.TestCase):
    def test_exclusions_overlap_without_double_counting(self) -> None:
        raw = pd.DataFrame({
            "text": ["NOT -5%", None, "\t ", "<b></b>", None, "Valid"],
            "sentiment": ["negative", "neutral", "positive", "neutral", None, pd.NA],
        }, index=[8, 8, 4, 3, 2, 1])
        snapshot = raw.copy(deep=True)
        processed, report = preprocess_news(raw)
        pd.testing.assert_frame_equal(raw, snapshot)
        self.assertEqual(list(processed.columns), ["source_row_id", "text", "sentiment", "text_clean"])
        self.assertEqual(processed["source_row_id"].tolist(), [0])
        self.assertEqual(processed["text_clean"].tolist(), ["not -5%"])
        self.assertEqual((report.n_input, report.n_retained, report.n_excluded), (6, 1, 5))
        self.assertEqual(report.n_input, report.n_retained + report.n_excluded)
        self.assertEqual(report.reason_counts, {
            "missing_text": 2, "missing_sentiment": 2,
            "blank_text": 1, "empty_after_cleaning": 1,
        })
        self.assertEqual([row.source_row_id for row in report.excluded_rows], [1, 2, 3, 4, 5])
        self.assertEqual(report.excluded_rows[3].reasons, ("missing_text", "missing_sentiment"))
        self.assertEqual(report.class_counts_before, {"negative": 1, "neutral": 2, "positive": 1})
        self.assertEqual(report.class_counts_after, {"negative": 1, "neutral": 0, "positive": 0})
        self.assertEqual(report.missing_sentiment_before, 2)
        self.assertEqual(report.changed_source_row_ids, (0,))

    def test_preserves_order_duplicates_literal_nan_and_extra_input(self) -> None:
        raw = pd.DataFrame({
            "text": ["same", " ", "same", "nan", "NA", "None"],
            "sentiment": ["neutral"] * 6, "provider": ["x"] * 6,
        }, index=[99, 55, 55, 12, 11, 10])
        processed, report = preprocess_news(raw)
        self.assertEqual(processed["source_row_id"].tolist(), [0, 2, 3, 4, 5])
        self.assertEqual(processed["text"].tolist(), ["same", "same", "nan", "NA", "None"])
        self.assertEqual(processed["text_clean"].tolist(), ["same", "same", "nan", "na", "none"])
        self.assertNotIn("provider", processed.columns)
        repeated, repeated_report = preprocess_news(raw)
        pd.testing.assert_frame_equal(processed, repeated)
        self.assertEqual(report, repeated_report)
        self.assertEqual(report.changed_source_row_ids, (4, 5))

    def test_contract_errors_not_silent_exclusions(self) -> None:
        cases = [
            pd.DataFrame({"text": ["news"]}),
            pd.DataFrame({"text": [], "sentiment": []}),
            pd.DataFrame({"text": [42], "sentiment": ["neutral"]}),
            pd.DataFrame({"text": ["news"], "sentiment": [42]}),
            pd.DataFrame({"text": ["news"], "sentiment": ["bullish"]}),
            pd.DataFrame({"text": ["news"], "sentiment": [" "]}),
            pd.DataFrame([["a", "b", "neutral"]], columns=["text", "text", "sentiment"]),
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    preprocess_news(raw)

    def test_all_excluded_error_keeps_diagnostics(self) -> None:
        raw = pd.DataFrame({"text": [None, "<b></b>"], "sentiment": [None, "neutral"]})
        with self.assertRaises(EmptyProcessedDatasetError) as caught:
            preprocess_news(raw)
        report = caught.exception.report
        self.assertEqual((report.n_input, report.n_retained, report.n_excluded), (2, 0, 2))
        self.assertEqual(report.reason_counts["missing_text"], 1)
        self.assertEqual(report.reason_counts["empty_after_cleaning"], 1)
        self.assertEqual(len(report.excluded_rows), 2)

    def test_compatibility_helper_uses_same_logic_and_reports_exclusions(self) -> None:
        raw = pd.DataFrame({"text": ["NOT up", None], "sentiment": ["negative", "neutral"]})
        expected, _ = preprocess_news(raw)
        with self.assertWarnsRegex(UserWarning, "Excluded 1 rows"):
            actual = clean_news_data(raw)
        pd.testing.assert_frame_equal(actual, expected)
        custom = pd.DataFrame({"headline": ["NOT up"], "sentiment": ["negative"]})
        self.assertEqual(clean_news_data(custom, text_col="headline")["text_clean"].tolist(), ["not up"])


if __name__ == "__main__":
    unittest.main()
