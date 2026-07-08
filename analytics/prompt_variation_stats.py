"""プロンプトバリエーション試行の平均・標準偏差を集計する

- 非推論: v1=008 既存結果, v2〜v5=012（Batch）
- 推論 medium: v1=010 既存結果, v2〜v5=013（Batch）

使い方（リポジトリルートで）:
  uv run analytics/prompt_variation_stats.py
"""

import json
import statistics
from pathlib import Path

import soramimi_phonetic_search_dataset
from soramimi_phonetic_search_dataset import load_default_dataset_for_llm

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "reproduce_leaderboard" / "results"
OUTPUT_PATH = Path(__file__).parent / "results" / "prompt_variation_stats.json"

TOPN = 10
VARIANTS = ["v1", "v2", "v3", "v4", "v5"]
SUBSET_KEYS = ["overall", "easy", "medium", "hard"]

GROUPS = {
    "non_reasoning": {
        "variation_dir": RESULTS_DIR / "012_prompt_variations",
        "prefix": "012",
        "conditions": {
            "simple": "008_01_llm_rerank_gpt54_simple.json",
            "detailed": "008_02_llm_rerank_gpt54_detailed.json",
            "step_by_step": "008_03_llm_rerank_gpt54_step_by_step.json",
            "detailed_romaji": "008_04_llm_rerank_gpt54_detailed_pyopenjtalk_romaji.json",
            "detailed_kana_romaji": "008_06_llm_rerank_gpt54_detailed_kana_and_pyopenjtalk_romaji.json",
        },
    },
    "reasoning_medium": {
        "variation_dir": RESULTS_DIR / "013_prompt_variations_reasoning",
        "prefix": "013",
        "conditions": {
            "simple": "010_01_llm_rerank_gpt54_medium_simple.json",
            "detailed": "010_02_llm_rerank_gpt54_medium_detailed.json",
            "step_by_step": "010_03_llm_rerank_gpt54_medium_step_by_step.json",
        },
    },
}


def load_subsets(queries: list[str]) -> list[str]:
    """難易度ラベルを生 JSON から読む（古い同梱データにない場合は GitHub 最新版）"""
    raw_path = (
        Path(soramimi_phonetic_search_dataset.__file__).parent / "data" / "baseball.json"
    )
    raw_queries = json.loads(raw_path.read_text(encoding="utf-8"))["queries"]
    if "subset" not in raw_queries[0]:
        cache_path = Path(__file__).parent / "results" / "baseball_with_subset.json"
        if not cache_path.exists():
            import urllib.request

            url = (
                "https://raw.githubusercontent.com/jiroshimaya/"
                "soramimi-phonetic-search-dataset/main/"
                "src/soramimi_phonetic_search_dataset/data/baseball.json"
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url) as response:
                cache_path.write_bytes(response.read())
        raw_queries = json.loads(cache_path.read_text(encoding="utf-8"))["queries"]
    if [q["query"] for q in raw_queries] != queries:
        raise ValueError("raw dataset order does not match loaded dataset order")
    return [q["subset"] for q in raw_queries]


def per_query_recalls(result_json: dict, subsets: list[str], queries: list[str]):
    results = result_json["results"]
    if len(results) != len(queries):
        raise ValueError(f"query count mismatch: {len(results)} != {len(queries)}")
    for row, query in zip(results, queries):
        if row["query"] != query:
            raise ValueError(f"query order mismatch: {row['query']} != {query}")
    recalls: dict[str, list[float]] = {key: [] for key in SUBSET_KEYS}
    for row, subset in zip(results, subsets):
        positives = set(row["positive_words"])
        hits = len(set(row["ranked_words"][:TOPN]) & positives)
        recall = hits / len(positives)
        recalls["overall"].append(recall)
        recalls[subset].append(recall)
    return {key: statistics.mean(values) for key, values in recalls.items()}


def main() -> None:
    dataset = load_default_dataset_for_llm(wordlist_size=100)
    queries = [q.query for q in dataset.queries]
    subsets = load_subsets(queries)

    summary: dict[str, dict] = {}
    for group_name, group in GROUPS.items():
        group_summary: dict[str, dict] = {}
        for condition, v1_file in group["conditions"].items():
            by_variant = {}
            for variant in VARIANTS:
                if variant == "v1":
                    path = RESULTS_DIR / v1_file
                else:
                    path = (
                        group["variation_dir"]
                        / f"{group['prefix']}_{condition}_{variant}.json"
                    )
                if not path.exists():
                    print(f"missing: {path}")
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                by_variant[variant] = per_query_recalls(data, subsets, queries)
            if not by_variant:
                continue
            stats = {}
            for key in SUBSET_KEYS:
                values = [v[key] for v in by_variant.values()]
                stats[key] = {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            group_summary[condition] = {"variants": by_variant, "stats": stats}
        summary[group_name] = group_summary

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"n = {len(queries)} queries, Recall@{TOPN}\n")
    for group_name, group_summary in summary.items():
        print(f"## {group_name}")
        header = (
            f"{'condition':<22} {'overall':>15} {'easy':>15} {'medium':>15} {'hard':>15}"
        )
        print(header)
        print("-" * len(header))
        for condition, entry in group_summary.items():
            cells = [
                f"{entry['stats'][key]['mean']:.3f}±{entry['stats'][key]['stdev']:.3f}"
                for key in SUBSET_KEYS
            ]
            n = entry["stats"]["overall"]["n"]
            print(
                f"{condition:<22} {cells[0]:>15} {cells[1]:>15} {cells[2]:>15} {cells[3]:>15}  (n={n})"
            )
        for condition, entry in group_summary.items():
            variant_strs = [
                f"{variant}={values['overall']:.3f}"
                for variant, values in entry["variants"].items()
            ]
            print(f"  {condition}: {' '.join(variant_strs)}")
        print()
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
