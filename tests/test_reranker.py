from pathlib import Path
import sys

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "reproduce_leaderboard"
        / "methods"
    ),
)

from common.reranker import transform_text_for_rerank


def test_transform_text_for_rerank_supports_kana_spaced():
    assert transform_text_for_rerank("リャオ", "kana_spaced") == "リ ャ オ"
    assert transform_text_for_rerank("ショー", "kana_spaced") == "シ ョ ー"
