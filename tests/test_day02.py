"""Offline minimal-v1 boundaries; fixtures are not notebook data."""

from pathlib import Path
import subprocess
import sys
from typing import cast
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finnews_sentiment.features.preprocess import preprocess_text


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


if __name__ == "__main__":
    unittest.main()
