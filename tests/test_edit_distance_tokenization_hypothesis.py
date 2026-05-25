import tiktoken

from analytics import edit_distance_tokenization_hypothesis as module


def test_build_segments_handles_spaced_and_unspaced():
    assert module.build_segments(["リ", "ャ", "オ"], spaced=False) == ["リ", "ャ", "オ"]
    assert module.build_segments(["リ", "ャ", "オ"], spaced=True) == ["リ", " ャ", " オ"]


def test_analyze_alignment_detects_strict_match_for_kana_only():
    encoding = tiktoken.get_encoding(module.ENCODING_NAME)
    stats = module.analyze_alignment(
        encoding,
        rendered_text="リャオ",
        units=["リ", "ャ", "オ"],
        spaced=False,
    )

    assert stats.strict_unit_token_match is True
    assert stats.boundary_aligned is True


def test_analyze_alignment_detects_misalignment_for_spaced_kana():
    encoding = tiktoken.get_encoding(module.ENCODING_NAME)
    stats = module.analyze_alignment(
        encoding,
        rendered_text="リ ャ オ",
        units=["リ", "ャ", "オ"],
        spaced=True,
    )

    assert stats.strict_unit_token_match is False
    assert stats.boundary_aligned is False
