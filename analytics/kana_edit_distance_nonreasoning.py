"""
LLM が reasoning なしでカタカナ文字列どうしの編集距離をどれくらい当てられるか測る小実験。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

from analytics.mora_edit_distance_nonreasoning import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    build_all_pairs,
    select_pairs,
    sequence_edit_distance,
    usage_to_dict,
)

DEFAULT_PROMPT_MODE = "kana_only"
KANA_PROMPT_MODES = ["kana_only", "kana_spaced"]


def kana_edit_distance(left: str, right: str) -> int:
    return sequence_edit_distance(list(left), list(right))


def build_system_prompt(prompt_mode: str) -> str:
    if prompt_mode == "kana_only":
        return """
あなたはカタカナ文字列どうしの編集距離を整数で返す判定器です。

- 比較対象は元のカタカナ文字列そのものです
- 1 文字の挿入・削除・置換のコストはすべて 1 とします
- 説明や途中経過は出さず、JSON object だけを返してください
- 出力は {"distance": 整数} のみ
""".strip()
    if prompt_mode == "kana_spaced":
        return """
あなたはスペース区切りのカタカナ文字列どうしの編集距離を整数で返す判定器です。

- 比較対象はスペース区切りで与えられる各要素です
- 1 要素の挿入・削除・置換のコストはすべて 1 とします
- たとえば「リ ャ オ」は 3 要素として扱います
- 説明や途中経過は出さず、JSON object だけを返してください
- 出力は {"distance": 整数} のみ
""".strip()
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def space_kana(text: str) -> str:
    return " ".join(list(text))


def build_user_prompt(pair, prompt_mode: str) -> str:
    if prompt_mode == "kana_only":
        return f"""
文字列A: {pair.query}
文字列B: {pair.candidate}

カタカナ文字列としての編集距離を整数で返してください。
""".strip()
    if prompt_mode == "kana_spaced":
        return f"""
文字列A: {space_kana(pair.query)}
文字列B: {space_kana(pair.candidate)}

スペース区切りの各要素どうしの編集距離を整数で返してください。
""".strip()
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def infer_distance(
    *,
    client: OpenAI,
    model_name: str,
    pair,
    max_output_tokens: int,
    reasoning_effort: str,
    prompt_mode: str,
) -> dict:
    from analytics.mora_edit_distance_nonreasoning import DistanceOutput

    request_kwargs: dict = {
        "model": model_name,
        "instructions": build_system_prompt(prompt_mode),
        "input": build_user_prompt(pair, prompt_mode),
        "text_format": DistanceOutput,
        "max_output_tokens": max_output_tokens,
    }
    if model_name.startswith("gpt-5"):
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.parse(**request_kwargs)
    parsed_output = response.output_parsed
    if parsed_output is None:
        raise ValueError(f"Structured output was empty for {model_name}")

    exact_distance = kana_edit_distance(pair.query, pair.candidate)
    predicted_distance = parsed_output.distance
    return {
        "pair_id": pair.pair_id,
        "query_index": pair.query_index,
        "candidate_index": pair.candidate_index,
        "query": pair.query,
        "candidate": pair.candidate,
        "exact_distance": exact_distance,
        "predicted_distance": predicted_distance,
        "absolute_error": abs(predicted_distance - exact_distance),
        "is_exact_match": predicted_distance == exact_distance,
        "prompt_mode": prompt_mode,
        "response_id": response.id,
        "usage": usage_to_dict(response),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compute_metrics(records: list[dict]) -> dict:
    exact_distances = [record["exact_distance"] for record in records]
    predicted_distances = [record["predicted_distance"] for record in records]
    absolute_errors = [record["absolute_error"] for record in records]
    exact_match_count = sum(1 for record in records if record["is_exact_match"])
    return {
        "pair_count": len(records),
        "exact_match_rate": exact_match_count / len(records),
        "mean_absolute_error": _mean(absolute_errors),
        "average_exact_distance": _mean(exact_distances),
        "average_predicted_distance": _mean(predicted_distances),
    }


def aggregate_usage(records: list[dict]) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "total_tokens": 0,
    }
    for record in records:
        usage = record.get("usage")
        if usage is None:
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
    return totals


def build_default_output_path(
    *, model_name: str, sample_size: int, seed: int, prompt_mode: str
) -> Path:
    model_name_safe = model_name.replace("/", "-")
    prompt_mode_suffix = (
        "" if prompt_mode == DEFAULT_PROMPT_MODE else f"_prompt{prompt_mode}"
    )
    return (
        Path(__file__).resolve().parent
        / "results"
        / (
            "kana_edit_distance_nonreasoning_"
            f"{model_name_safe}{prompt_mode_suffix}_small_sample{sample_size}_seed{seed}.json"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="small_dataset_for_llm の pair を使って、LLM が reasoning なしでカナ編集距離をどれだけ当てられるかを測る"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--wordlist_size", type=int, default=100)
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="none",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--max_output_tokens", type=int, default=64)
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="kana_only",
        choices=KANA_PROMPT_MODES,
    )
    parser.add_argument("-o", "--output_file_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_pairs = build_all_pairs(wordlist_size=args.wordlist_size)
    sampled_pairs = select_pairs(all_pairs, sample_size=args.sample_size, seed=args.seed)
    output_file_path = args.output_file_path or build_default_output_path(
        model_name=args.model,
        sample_size=args.sample_size,
        seed=args.seed,
        prompt_mode=args.prompt_mode,
    )
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    records = [
        infer_distance(
            client=client,
            model_name=args.model,
            pair=pair,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            prompt_mode=args.prompt_mode,
        )
        for pair in sampled_pairs
    ]

    output = {
        "parameters": {
            "model": args.model,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "wordlist_size": args.wordlist_size,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "prompt_mode": args.prompt_mode,
            "output_file_path": str(output_file_path),
        },
        "dataset": {
            "name": "small_dataset_for_llm",
            "total_pair_count": len(all_pairs),
        },
        "metrics": {
            **compute_metrics(records),
            "usage": aggregate_usage(records),
        },
        "pairs": records,
    }

    with output_file_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
