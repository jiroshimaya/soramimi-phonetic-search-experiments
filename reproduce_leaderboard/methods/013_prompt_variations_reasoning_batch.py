"""推論あり（medium）条件のプロンプトバリエーション追加試行（v2〜v5）を OpenAI Batch API で実行するスクリプト

使い方（リポジトリルートで）:
  uv run reproduce_leaderboard/methods/013_prompt_variations_reasoning_batch.py preview
  uv run reproduce_leaderboard/methods/013_prompt_variations_reasoning_batch.py submit
  uv run reproduce_leaderboard/methods/013_prompt_variations_reasoning_batch.py status
  uv run reproduce_leaderboard/methods/013_prompt_variations_reasoning_batch.py retrieve

条件は 010_01〜010_03 の full 評価（推論 medium 3条件）に対応し、各条件について
prompt_variations.py の v2〜v5 を1試行ずつ実行する（v1 は既存の 010_* 結果）。
"""

import argparse
import json
import sys
from pathlib import Path

METHODS_DIR = Path(__file__).parent
sys.path.insert(0, str(METHODS_DIR / "common"))

import reranker  # noqa: E402
from prompt_variations import PROMPT_VARIATIONS  # noqa: E402
from reranker import (  # noqa: E402
    OPENAI_BATCH_DISCOUNT_FACTOR,
    build_rerank_messages,
    calculate_token_cost,
    get_rerank_response_format,
    retrieve_openai_batch_rerank_job,
    submit_openai_batch_rerank_job,
    transform_text_for_rerank,
)
from soramimi_phonetic_search_dataset import (  # noqa: E402
    RankingFunctionOutput,
    evaluate_ranking_function,
    load_default_dataset_for_llm,
)

TOPN = 10
RERANK_INPUT_SIZE = 100
MODEL_NAME = "gpt-5.4"
REASONING_EFFORT = "medium"
VARIANTS = ["v2", "v3", "v4", "v5"]

EXAMPLE_QUERY = "タロウ"
EXAMPLE_WORDLIST = ["アオ", "アオウヅ", "アノウ", "タキョウ", "タド", "タノ", "タロウ", "タンノ"]
EXAMPLE_TOPN = 5
EXAMPLE_RERANKED = [6, 4, 5, 7, 2]

# 010_* の full 評価3条件（推論 medium）に対応
CONDITIONS = {
    "simple": {
        "prompt_key": "simple",
        "input_transform": "none",
        "rank_func": "llm_rerank",
        "baseline_script": "010_01",
    },
    "detailed": {
        "prompt_key": "detailed",
        "input_transform": "none",
        "rank_func": "llm_rerank",
        "baseline_script": "010_02",
    },
    "step_by_step": {
        "prompt_key": "step_by_step",
        "input_transform": "none",
        "rank_func": "llm_rerank",
        "baseline_script": "010_03",
    },
}

OUTPUT_DIR = METHODS_DIR.parent / "results" / "013_prompt_variations_reasoning"


def build_example_suffix(input_transform: str) -> str:
    example_wordlist = "\n".join(
        f"{index}. {transform_text_for_rerank(word, input_transform)}"
        for index, word in enumerate(EXAMPLE_WORDLIST)
    )
    reranked = ", ".join(str(index) for index in EXAMPLE_RERANKED)
    return f"""
Example:
Query: {transform_text_for_rerank(EXAMPLE_QUERY, input_transform)}
Wordlist:
{example_wordlist}
Top N: {EXAMPLE_TOPN}
Reranked: {reranked}
"""


def state_path_for(condition: str, variant: str) -> Path:
    return OUTPUT_DIR / f"013_{condition}_{variant}_openai_batch_state.json"


def result_path_for(condition: str, variant: str) -> Path:
    return OUTPUT_DIR / f"013_{condition}_{variant}.json"


def iter_jobs(conditions: list[str] | None, variants: list[str] | None):
    for condition, config in CONDITIONS.items():
        if conditions and condition not in conditions:
            continue
        for variant in VARIANTS:
            if variants and variant not in variants:
                continue
            yield condition, config, variant


def cmd_preview(args) -> None:
    dataset = load_default_dataset_for_llm(wordlist_size=RERANK_INPUT_SIZE)
    query = dataset.queries[0]
    for condition, config, variant in iter_jobs(args.conditions, args.variants):
        instructions = PROMPT_VARIATIONS[config["prompt_key"]][variant]
        messages = build_rerank_messages(
            [query.query],
            [query.wordlist],
            topn=TOPN,
            prompt_template=config["prompt_key"],
            prompt_instructions=instructions,
            prompt_example_suffix=build_example_suffix(config["input_transform"]),
            input_transform=config["input_transform"],
        )
        print("=" * 70)
        print(f"[{condition} / {variant}] input_transform={config['input_transform']}")
        print("-" * 30, "system", "-" * 30)
        print(messages[0][0]["content"])
        print("-" * 30, "user (先頭600字)", "-" * 24)
        print(messages[0][1]["content"][:600])
        print()


def cmd_submit(args) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_default_dataset_for_llm(wordlist_size=RERANK_INPUT_SIZE)
    query_texts = [q.query for q in dataset.queries]
    wordlists = [q.wordlist for q in dataset.queries]
    positives = [q.positive_words for q in dataset.queries]
    response_format = get_rerank_response_format(include_thoughts=False)

    submitted = []
    for condition, config, variant in iter_jobs(args.conditions, args.variants):
        state_path = state_path_for(condition, variant)
        if state_path.exists() and not args.force:
            print(f"skip (state exists): {condition}/{variant}")
            continue
        instructions = PROMPT_VARIATIONS[config["prompt_key"]][variant]
        state = submit_openai_batch_rerank_job(
            query_texts=query_texts,
            wordlist_texts=wordlists,
            positive_texts=positives,
            topn=TOPN,
            model_name=MODEL_NAME,
            prompt_template=config["prompt_key"],
            prompt_instructions=instructions,
            prompt_example_suffix=build_example_suffix(config["input_transform"]),
            input_transform=config["input_transform"],
            response_format=response_format,
            state_path=str(state_path),
            output_file_path=str(result_path_for(condition, variant)),
            reasoning_effort=REASONING_EFFORT,
        )
        submitted.append(
            {
                "condition": condition,
                "variant": variant,
                "batch_id": state["batch_id"],
                "status": state["batch_status"],
            }
        )
        print(f"submitted: {condition}/{variant} batch_id={state['batch_id']}")

    summary_path = OUTPUT_DIR / "submitted_batches.json"
    existing = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    existing.extend(submitted)
    summary_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{len(submitted)} 件のバッチを投入しました（一覧: {summary_path}）")


def cmd_status(args) -> None:
    from openai import OpenAI

    client = OpenAI()
    for condition, _config, variant in iter_jobs(args.conditions, args.variants):
        state_path = state_path_for(condition, variant)
        if not state_path.exists():
            print(f"{condition}/{variant}: 未投入")
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        batch = client.batches.retrieve(state["batch_id"])
        counts = getattr(batch, "request_counts", None)
        counts_str = (
            f" ({counts.completed}/{counts.total} done, {counts.failed} failed)"
            if counts
            else ""
        )
        print(f"{condition}/{variant}: {batch.status}{counts_str}")


def cmd_retrieve(args) -> None:
    dataset = load_default_dataset_for_llm(wordlist_size=RERANK_INPUT_SIZE)
    response_format = get_rerank_response_format(include_thoughts=False)

    for condition, config, variant in iter_jobs(args.conditions, args.variants):
        state_path = state_path_for(condition, variant)
        result_path = result_path_for(condition, variant)
        if not state_path.exists():
            print(f"{condition}/{variant}: 未投入のためスキップ")
            continue
        if result_path.exists() and not args.force:
            print(f"{condition}/{variant}: 結果ファイルが既に存在するためスキップ")
            continue
        try:
            retrieved = retrieve_openai_batch_rerank_job(
                state_path=str(state_path),
                response_format=response_format,
            )
        except RuntimeError as exc:
            print(f"{condition}/{variant}: {exc}")
            continue

        def ranking_func(
            query_texts: list[str], wordlists: list[list[str]]
        ) -> RankingFunctionOutput:
            return RankingFunctionOutput(
                ranked_wordlists=retrieved.reranked_wordlists,
            )

        results = evaluate_ranking_function(
            ranking_func=ranking_func,
            topn=TOPN,
            dataset=dataset,
        )
        results.metrics.execution_time = retrieved.execution_time
        results.parameters.rank_func = config["rank_func"]
        results.parameters.metadata.update(
            {
                "vowel_ratio": 0.5 if config["rank_func"] == "vowel_consonant" else None,
                "rerank": True,
                "rerank_model_name": MODEL_NAME,
                "rerank_reasoning_effort": REASONING_EFFORT,
                "rerank_prompt_template": config["prompt_key"],
                "rerank_prompt_variant": variant,
                "rerank_prompt_instructions": PROMPT_VARIATIONS[config["prompt_key"]][
                    variant
                ].strip(),
                "rerank_prompt_example_suffix": build_example_suffix(
                    config["input_transform"]
                ).strip(),
                "rerank_input_transform": config["input_transform"],
                "rerank_input_size": RERANK_INPUT_SIZE,
                "rerank_backend": "openai_batch",
                "rerank_batch_id": retrieved.batch_id,
                "baseline_script": config["baseline_script"],
            }
        )

        token_usage = reranker.get_last_token_usage()
        metrics_metadata = {
            "model_name": MODEL_NAME,
            "token_usage": {
                "input_tokens": token_usage.input_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "reasoning_tokens": token_usage.reasoning_tokens,
                "total_tokens": token_usage.total_tokens,
            },
        }
        try:
            token_cost = calculate_token_cost(
                MODEL_NAME, token_usage, discount_factor=OPENAI_BATCH_DISCOUNT_FACTOR
            )
            metrics_metadata["cost"] = {
                "input_cost": token_cost.input_cost,
                "output_cost": token_cost.output_cost,
                "reasoning_cost": token_cost.reasoning_cost,
                "total_cost": token_cost.total_cost,
                "discount_factor": OPENAI_BATCH_DISCOUNT_FACTOR,
            }
        except Exception as exc:  # litellm が料金を知らない場合など
            metrics_metadata["cost_error"] = str(exc)
        results.metrics.metadata = metrics_metadata

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2,
                default=lambda x: x.__dict__,
            )
        print(
            f"{condition}/{variant}: Recall@{TOPN} = {results.metrics.recall:.4f} "
            f"-> {result_path.name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in [
        ("preview", cmd_preview),
        ("submit", cmd_submit),
        ("status", cmd_status),
        ("retrieve", cmd_retrieve),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--conditions", nargs="*", choices=list(CONDITIONS))
        p.add_argument("--variants", nargs="*", choices=VARIANTS)
        p.add_argument("--force", action="store_true")
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
