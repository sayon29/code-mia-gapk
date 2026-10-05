import pytest

from code_mia.data.canonicalize import canonicalize_prefix, word_spans


@pytest.mark.parametrize("prefix", [
    "# comment\n\n    def café(x):\n        return x + 1  # keep punctuation!\n",
    "\timport os\n\n# licence: MIT\nvalue = '🙂'\n",
    "same same same; same(same)\n",
])
def test_canonical_text_is_exact_prefix_with_structure(prefix):
    raw = prefix + " ".join(f"token{i}" for i in range(60))
    result = canonicalize_prefix(raw, 32)
    assert result.eligible
    assert raw.startswith(result.text)
    assert result.text == raw[: word_spans(raw)[31][1]]
    assert result.text.encode() == raw.encode()[: len(result.text.encode())]


def test_null_empty_and_short_reasons():
    assert canonicalize_prefix(None).exclusion_reason == "null_content"
    assert canonicalize_prefix(" \n\t").exclusion_reason == "empty_content"
    assert canonicalize_prefix("only two").exclusion_reason == "fewer_than_32_nltk_words"

