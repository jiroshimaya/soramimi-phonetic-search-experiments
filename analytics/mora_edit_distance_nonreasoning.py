"""
LLM が reasoning なしでモーラ列の編集距離をどれくらい当てられるか測る小実験。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jamorasep
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soramimi_phonetic_search_dataset import (  # noqa: E402
    DEFAULT_LLM_WORDLIST_SIZE,
    SMALL_DATASET_QUERY_COUNT,
    load_default_dataset_for_llm,
)

try:  # noqa: E402
    from soramimi_phonetic_search_dataset import (
        load_small_dataset_for_llm as _package_load_small_dataset_for_llm,
    )
except ImportError:  # pragma: no cover - current pinned dataset does not export this.
    _package_load_small_dataset_for_llm = None


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_SAMPLE_SIZE = 3
DEFAULT_SEED = 7


class DistanceOutput(BaseModel):
    distance: int = Field(
        description="Two mora sequences' Levenshtein edit distance as a non-negative integer."
    )


@dataclass(frozen=True)
class MoraDistancePair:
    pair_id: str
    query_index: int
    candidate_index: int
    query: str
    candidate: str
    query_moras: list[str]
    candidate_moras: list[str]
    exact_distance: int
    is_positive: bool


def load_small_dataset_for_llm(*, wordlist_size: int = DEFAULT_LLM_WORDLIST_SIZE):
    if _package_load_small_dataset_for_llm is not None:
        return _package_load_small_dataset_for_llm(wordlist_size=wordlist_size)
    return load_default_dataset_for_llm(
        query_limit=SMALL_DATASET_QUERY_COUNT,
        wordlist_size=wordlist_size,
    )


def to_moras(text: str) -> list[str]:
    return jamorasep.parse(text)


def sequence_edit_distance(left: list[str], right: list[str]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            current_row.append(
                min(
                    previous_row[right_index] + 1,
                    current_row[right_index - 1] + 1,
                    previous_row[right_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]


def mora_edit_distance(left: str, right: str) -> int:
    return sequence_edit_distance(to_moras(left), to_moras(right))


def build_all_pairs(*, wordlist_size: int) -> list[MoraDistancePair]:
    dataset = load_small_dataset_for_llm(wordlist_size=wordlist_size)
    pairs: list[MoraDistancePair] = []
    for query_index, query_with_wordlist in enumerate(dataset.queries):
        positive_words = set(query_with_wordlist.positive_words)
        query_moras = to_moras(query_with_wordlist.query)
        for candidate_index, candidate in enumerate(query_with_wordlist.wordlist):
            candidate_moras = to_moras(candidate)
            pairs.append(
                MoraDistancePair(
                    pair_id=f"q{query_index:02d}_c{candidate_index:03d}",
                    query_index=query_index,
                    candidate_index=candidate_index,
                    query=query_with_wordlist.query,
                    candidate=candidate,
                    query_moras=query_moras,
                    candidate_moras=candidate_moras,
                    exact_distance=sequence_edit_distance(
                        query_moras,
                        candidate_moras,
                    ),
                    is_positive=candidate in positive_words,
                )
            )
    return pairs


def select_pairs(
    pairs: list[MoraDistancePair],
    *,
    sample_size: int,
    seed: int,
) -> list[MoraDistancePair]:
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if sample_size > len(pairs):
        raise ValueError(
            f"sample_size must be <= total pair count ({len(pairs)}), got {sample_size}"
        )
    selected = random.Random(seed).sample(pairs, sample_size)
    return sorted(selected, key=lambda pair: (pair.query_index, pair.candidate_index))


def build_system_prompt() -> str:
    return """
あなたはモーラ列どうしの編集距離を整数で返す判定器です。

- 比較対象はスペース区切りで与えられるモーラ列です
- 1 モーラの挿入・削除・置換のコストはすべて 1 とします
- 説明や途中経過は出さず、JSON object だけを返してください
- 出力は {"distance": 整数} のみ
""".strip()


def build_user_prompt(pair: MoraDistancePair) -> str:
    return f"""
元の文字列A: {pair.query}
元の文字列B: {pair.candidate}
モーラ列A: {' '.join(pair.query_moras)}
モーラ列B: {' '.join(pair.candidate_moras)}

モーラ編集距離を整数で返してください。
""".strip()


def usage_to_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    output_tokens_details = getattr(usage, "output_tokens_details", None)
    input_tokens_details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": getattr(output_tokens_details, "reasoning_tokens", None),
        "cached_input_tokens": getattr(input_tokens_details, "cached_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def infer_distance(
    *,
    client: OpenAI,
    model_name: str,
    pair: MoraDistancePair,
    max_output_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": build_system_prompt(),
        "input": build_user_prompt(pair),
        "text_format": DistanceOutput,
        "max_output_tokens": max_output_tokens,
    }
    if model_name.startswith("gpt-5"):
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.parse(**request_kwargs)
    parsed_output = response.output_parsed
    if parsed_output is None:
        raise ValueError(f"Structured output was empty for {model_name}")

    predicted_distance = parsed_output.distance
    absolute_error = abs(predicted_distance - pair.exact_distance)
    return {
        **asdict(pair),
        "predicted_distance": predicted_distance,
        "absolute_error": absolute_error,
        "is_exact_match": predicted_distance == pair.exact_distance,
        "response_id": response.id,
        "usage": usage_to_dict(response),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def pearson_correlation(left: list[int], right: list[int]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Both input lists must have the same length")
    if len(left) < 2:
        return None

    left_mean = _mean(left)
    right_mean = _mean(right)
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    if left_variance == 0 or right_variance == 0:
        return None
    return covariance / ((left_variance * right_variance) ** 0.5)


def aggregate_usage(records: list[dict[str, Any]]) -> dict[str, int]:
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


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")

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
        "pearson_correlation": pearson_correlation(
            exact_distances,
            predicted_distances,
        ),
    }


def build_default_output_path(*, model_name: str, sample_size: int, seed: int) -> Path:
    model_name_safe = model_name.replace("/", "-")
    return (
        Path(__file__).resolve().parent
        / "results"
        / (
            "mora_edit_distance_nonreasoning_"
            f"{model_name_safe}_small_sample{sample_size}_seed{seed}.json"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "small_dataset_for_llm を使って、LLM が reasoning なしで "
            "モーラ編集距離をどれだけ当てられるかを測る"
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"使用するモデル名。デフォルト: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"評価する pair 数。デフォルト: {DEFAULT_SAMPLE_SIZE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"pair サンプリング用の seed。デフォルト: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--wordlist_size",
        type=int,
        default=DEFAULT_LLM_WORDLIST_SIZE,
        help="各 query で使う候補語数。デフォルト: 100",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="none",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="gpt-5 系に渡す reasoning.effort。デフォルト: none",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=64,
        help="Responses API の max_output_tokens。デフォルト: 64",
    )
    parser.add_argument(
        "-o",
        "--output_file_path",
        type=Path,
        help="結果 JSON の保存先。省略時は analytics/results/ 以下へ保存",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_pairs = build_all_pairs(wordlist_size=args.wordlist_size)
    sampled_pairs = select_pairs(
        all_pairs,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    output_file_path = args.output_file_path or build_default_output_path(
        model_name=args.model,
        sample_size=args.sample_size,
        seed=args.seed,
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
            "output_file_path": str(output_file_path),
        },
        "dataset": {
            "name": "small_dataset_for_llm",
            "query_count": SMALL_DATASET_QUERY_COUNT,
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
