"""LLMリランク (gpt-5.4, detailed prompt, kana-spaced input, small dataset_for_llm) による評価を実行するスクリプト"""

import json
import os
from pathlib import Path

from common.reranker import calculate_token_cost, get_last_token_usage, rank_by_llm
from soramimi_phonetic_search_dataset import (
    SMALL_DATASET_QUERY_COUNT,
    RankingFunctionOutput,
    evaluate_ranking_function,
    load_default_dataset_for_llm,
)

TOPN = 10
RERANK_INPUT_SIZE = 100
RERANK_BATCH_SIZE = 10
RERANK_INTERVAL = 0
MODEL_NAME = "gpt-5.4"
VOWEL_RATIO = 0.5
PROMPT_TEMPLATE = "detailed"
INPUT_TRANSFORM = "kana_spaced"
PROMPT_INSTRUCTIONS = """
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
- Query と Wordlist のカタカナは 1 文字ずつスペース区切りで与えられます
- 比較するときは、この区切りをカナ単位の切れ目として使ってください
- 子音より母音の一致を優先してください
- クエリと文字数が同じであることを優先してください。ただし促音（ッ）、撥音（ン）、長音（ー）の挿入や削除はある程度許容されます
出力は上位Top N件のインデックスのみ返してください。
"""
PROMPT_EXAMPLE_SUFFIX = """
Example:
Query: タ ロ ウ
Wordlist:
0. ア オ
1. ア オ ウ ヅ
2. ア ノ ウ
3. タ キ ョ ウ
4. タ ド
5. タ ノ
6. タ ロ ウ
7. タ ン ノ
Top N: 5
Reranked: 6, 4, 5, 7, 2
"""


def kana_spaced(text: str) -> str:
    return " ".join(text)


def strip_spaces(text: str) -> str:
    return text.replace(" ", "")


def build_rerank_metrics_metadata(model_name: str) -> dict[str, object]:
    token_usage = get_last_token_usage()
    token_cost = calculate_token_cost(model_name, token_usage)
    return {
        "model_name": model_name,
        "token_usage": {
            "input_tokens": token_usage.input_tokens,
            "output_tokens": token_usage.output_tokens,
            "reasoning_tokens": token_usage.reasoning_tokens,
            "total_tokens": token_usage.total_tokens,
        },
        "cost": {
            "input_cost": token_cost.input_cost,
            "output_cost": token_cost.output_cost,
            "reasoning_cost": token_cost.reasoning_cost,
            "total_cost": token_cost.total_cost,
        },
    }


def build_ranking_function():
    def ranking_func(
        query_texts: list[str],
        wordlists: list[list[str]],
    ) -> RankingFunctionOutput:
        spaced_queries = [kana_spaced(query) for query in query_texts]
        spaced_wordlists = [
            [kana_spaced(word) for word in wordlist] for wordlist in wordlists
        ]
        ranked_wordlists = rank_by_llm(
            query_texts=spaced_queries,
            wordlist_texts=spaced_wordlists,
            topn=TOPN,
            model_name=MODEL_NAME,
            reasoning_effort="none",
            prompt_template=PROMPT_TEMPLATE,
            prompt_instructions=PROMPT_INSTRUCTIONS,
            prompt_example_suffix=PROMPT_EXAMPLE_SUFFIX,
            input_transform="none",
            batch_size=RERANK_BATCH_SIZE,
            rerank_interval=RERANK_INTERVAL,
        )
        normalized_ranked_wordlists = [
            [strip_spaces(word) for word in ranked_wordlist]
            for ranked_wordlist in ranked_wordlists
        ]
        return RankingFunctionOutput(
            ranked_wordlists=normalized_ranked_wordlists,
            metrics_metadata=build_rerank_metrics_metadata(MODEL_NAME),
        )

    return ranking_func


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to run this experiment.")

    output_dir = Path(__file__).parent.parent / "results_small"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "008_08_llm_rerank_gpt54_detailed_kana_spaced.json"

    dataset = load_default_dataset_for_llm(
        query_limit=SMALL_DATASET_QUERY_COUNT,
        wordlist_size=RERANK_INPUT_SIZE,
    )
    results = evaluate_ranking_function(
        ranking_func=build_ranking_function(),
        topn=TOPN,
        dataset=dataset,
    )

    results.parameters.rank_func = "vowel_consonant"
    results.parameters.metadata.update(
        {
            "vowel_ratio": VOWEL_RATIO,
            "rerank": True,
            "rerank_backend": "litellm",
            "rerank_input_size": RERANK_INPUT_SIZE,
            "rerank_batch_size": RERANK_BATCH_SIZE,
            "rerank_interval": RERANK_INTERVAL,
            "rerank_model_name": MODEL_NAME,
            "rerank_reasoning_effort": "none",
            "rerank_prompt_template": PROMPT_TEMPLATE,
            "rerank_prompt_instructions": PROMPT_INSTRUCTIONS.strip(),
            "rerank_prompt_example_suffix": PROMPT_EXAMPLE_SUFFIX.strip(),
            "rerank_input_transform": INPUT_TRANSFORM,
            "dataset_size": "small",
            "dataset_loader": "small_dataset_for_llm",
        }
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.__dict__,
        )

    print("Recall: ", results.metrics.recall)
    print("Execution time: ", results.metrics.execution_time)


if __name__ == "__main__":
    main()
