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
    """The compressed bytes should start with the documented magic+version+lang
    +crc+token_count header."""
    blob = compress_text("Hello world.", verbose=False)
    assert blob[:4] == b"NNZP", "missing magic bytes"
    assert blob[4] == 3, "wrong file format version"
    # next 2 bytes are the lang code, ascii
    lang = blob[5:7].decode("ascii").rstrip()
    assert lang == "en", f"expected en for English input, got {lang!r}"


def test_integrity_check_catches_corrupted_crc_field():
    """If the CRC32 stored in the header doesn't match the decompressed
    text's CRC32, decompress_bytes must raise IntegrityError. We simulate
    this by flipping a bit in the header's CRC field — the payload still
    decodes fine, but the stored CRC is now wrong."""
    from nnzip import IntegrityError, decompress_bytes

    text = "The quick brown fox jumps over the lazy dog."
    blob = compress_text(text, verbose=False)

    # v3 header layout: magic(4) + version(1) + lang(2) + crc(4) + count(4)
    # so bytes 7..10 are the CRC. Flip one bit there.
    corrupted = bytearray(blob)
    corrupted[7] ^= 0x01
    corrupted = bytes(corrupted)

    with pytest.raises(IntegrityError):
        decompress_bytes(corrupted, verbose=False)


def test_explicit_lang_override():
    """Passing lang='en' explicitly should produce the same result as
    autodetect on English text."""
    text = "Hello world. This is plain English."
    blob_auto = compress_text(text, verbose=False)
    blob_explicit = compress_text(text, lang="en", verbose=False)
    assert blob_auto == blob_explicit
