"""Prefix-preserving canonicalization for the controlled NLTK-word condition."""
from __future__ import annotations

from dataclasses import dataclass

from nltk.tokenize import TreebankWordTokenizer


@dataclass(frozen=True)
class CanonicalizedText:
    text: str
    word_count: int
    eligible: bool
    exclusion_reason: str | None


def word_spans(text: str) -> list[tuple[int, int]]:
    """NLTK Treebank spans, equivalent to word_tokenize(..., preserve_line=True)."""
    return list(TreebankWordTokenizer().span_tokenize(text))


def canonicalize_prefix(raw_text: str | None, word_units: int = 32) -> CanonicalizedText:
    if raw_text is None:
        return CanonicalizedText("", 0, False, "null_content")
    if not isinstance(raw_text, str):
        return CanonicalizedText("", 0, False, "non_string_content")
    if raw_text == "" or raw_text.strip() == "":
        return CanonicalizedText("", 0, False, "empty_content")
    spans = word_spans(raw_text)
    if len(spans) < word_units:
        return CanonicalizedText("", len(spans), False, f"fewer_than_{word_units}_nltk_words")
    prefix = raw_text[: spans[word_units - 1][1]]
    if not raw_text.startswith(prefix):  # defensive invariant
        raise AssertionError("Canonicalization did not produce an exact prefix")
    return CanonicalizedText(prefix, len(spans), True, None)

