"""kanasim 0.0.11 で追加された距離オプションの変種を full dataset で評価するスクリプト

kanasim に normalize / symmetric / phoneme_unit / consonant_distance /
vowel_binary の各オプションが追加されたので（kanasim PR #8〜#13）、
leaderboard の KanaSim EditDistance (vowel_ratio=0.8, Recall@10=0.831) を
基準に、変種の Recall@10 を比較する。LLM は使わない。

使い方（リポジトリルートで）:
  uv run reproduce_leaderboard/methods/014_kanasim_variants.py

結果は reproduce_leaderboard/results/014_kanasim_variants.json に保存される
（per-query の ranked_words は保存せず、変種ごとの recall のみ）。
"""

import datetime
import functools
import json
from importlib.metadata import version
from pathlib import Path

from soramimi_phonetic_search_dataset import (
    evaluate_ranking_function,
    rank_by_kanasim,
)

TOPN = 10
OUTPUT_PATH = (
    Path(__file__).parent.parent / "results" / "014_kanasim_variants.json"
)

# (変種名, create_kana_distance_calculator に渡す kwargs)
VARIANTS: list[tuple[str, dict]] = [
    # 基準（leaderboard 掲載の設定）
    ("baseline vr0.8", dict(vowel_ratio=0.8)),
    # vowel_ratio 感度（生スケール）
    ("baseline vr0.6", dict(vowel_ratio=0.6)),
    ("baseline vr0.7", dict(vowel_ratio=0.7)),
    ("baseline vr0.9", dict(vowel_ratio=0.9)),
    # 対称化
    ("symmetric vr0.8", dict(vowel_ratio=0.8, symmetric=True)),
    # 正規化（子音・母音距離を各[0,1]化。最適 vowel_ratio が生スケールと変わる）
    ("normalize vr0.6", dict(vowel_ratio=0.6, normalize=True)),
    ("normalize vr0.7", dict(vowel_ratio=0.7, normalize=True)),
    ("normalize vr0.8", dict(vowel_ratio=0.8, normalize=True)),
    ("normalize vr0.9", dict(vowel_ratio=0.9, normalize=True)),
    ("normalize+symmetric vr0.8", dict(vowel_ratio=0.8, normalize=True, symmetric=True)),
    # 母音バイナリ（押韻向け設定。normalize 必須、低 vowel_ratio が釣り合う）
    ("vowel_binary+normalize vr0.3", dict(vowel_ratio=0.3, vowel_binary=True, normalize=True)),
    ("vowel_binary+normalize vr0.4", dict(vowel_ratio=0.4, vowel_binary=True, normalize=True)),
    ("vowel_binary+normalize vr0.5", dict(vowel_ratio=0.5, vowel_binary=True, normalize=True)),
    ("vowel_binary+normalize vr0.6", dict(vowel_ratio=0.6, vowel_binary=True, normalize=True)),
    ("vowel_binary+normalize vr0.8", dict(vowel_ratio=0.8, vowel_binary=True, normalize=True)),
    ("vowel_binary+normalize+symmetric vr0.8", dict(vowel_ratio=0.8, vowel_binary=True, normalize=True, symmetric=True)),
    # モノフォン距離表（バイフォン平均）
    ("mono_avg vr0.8", dict(vowel_ratio=0.8, phoneme_unit="mono")),
    ("mono_avg vowel_binary+normalize vr0.8", dict(vowel_ratio=0.8, phoneme_unit="mono", vowel_binary=True, normalize=True)),
    # 弁別素性ベースの子音距離
    ("features+normalize vr0.6", dict(vowel_ratio=0.6, phoneme_unit="mono", consonant_distance="features", normalize=True)),
    ("features+normalize vr0.8", dict(vowel_ratio=0.8, phoneme_unit="mono", consonant_distance="features", normalize=True)),
    ("features vowel_binary+normalize vr0.8", dict(vowel_ratio=0.8, phoneme_unit="mono", consonant_distance="features", vowel_binary=True, normalize=True)),
]


def main() -> None:
    kanasim_version = version("kanasim")
    if tuple(int(x) for x in kanasim_version.split(".")[:3]) < (0, 0, 11):
        raise SystemExit(
            f"kanasim>=0.0.11 が必要です (installed: {kanasim_version})"
        )

    results = []
    for name, kwargs in VARIANTS:
        r = evaluate_ranking_function(
            ranking_func=functools.partial(rank_by_kanasim, **kwargs), topn=TOPN
        )
        row = {
            "variant": name,
            "kanasim_kwargs": kwargs,
            "recall": r.metrics.recall,
            "execution_time": r.metrics.execution_time,
        }
        results.append(row)
        print(f"{name}: recall@{TOPN}={r.metrics.recall:.3f}", flush=True)

    output = {
        "parameters": {
            "topn": TOPN,
            "rank_func": "kanasim",
            "kanasim_version": kanasim_version,
            "execution_timestamp": datetime.datetime.now().isoformat(),
        },
        "results": results,
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
