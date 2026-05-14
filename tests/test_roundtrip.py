"""Round-trip tests: every text that goes through compress+decompress should
come out byte-identical. This is the bare minimum we want to be confident in
before each release."""

import pytest

from nnzip import compress_text, decompress_bytes


CASES = [
    ("short", "The cat sat on the mat."),
    ("punctuation", 'She said, "It\'s a lovely day, isn\'t it?"'),
    ("multi_sentence",
     "First sentence. Second one follows. And a third, for good measure."),
    ("with_newlines", "Line one.\nLine two.\nLine three.\n"),
    ("longer_paragraph",
     "The morning rain pattered against the windows of the small cottage. "
     "Margaret stirred her tea slowly, watching the steam rise and curl in "
     "the cool air. The fire in the hearth had burned low overnight."),
    ("repeated_phrase",
     "Hello world. Hello world. Hello world. Hello world. Hello world."),
]


@pytest.mark.parametrize("name,text", CASES, ids=[c[0] for c in CASES])
def test_round_trip(name, text):
    blob = compress_text(text, verbose=False)
    recovered = decompress_bytes(blob, verbose=False)
    assert recovered == text, f"round-trip mismatch for case {name!r}"


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="empty"):
        compress_text("", verbose=False)


def test_file_format_header():
    """The compressed bytes should start with the magic+version+token_count
    header we documented."""
    blob = compress_text("Hello world.", verbose=False)
    assert blob[:4] == b"NNZP", "missing magic bytes"
    assert blob[4] == 2, "wrong file format version"
