"""
LLM が reasoning なしでカタカナ文字列をモーラ列へどれくらい正確に分割できるか測る小実験。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analytics.mora_edit_distance_nonreasoning import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    load_small_dataset_for_llm,
    to_moras,
    usage_to_dict,
)


class MoraSegmentationOutput(BaseModel):
    moras: list[str] = Field(
        description="The input katakana string segmented into moras."
    )


@dataclass(frozen=True)
class MoraSegmentationSample:
    sample_id: str
    text: str
    gold_moras: list[str]
    source_type: str
    query_index: int
    candidate_index: int | None = None


def collect_unique_strings(*, wordlist_size: int) -> list[MoraSegmentationSample]:
    dataset = load_small_dataset_for_llm(wordlist_size=wordlist_size)
    seen_texts: set[str] = set()
    samples: list[MoraSegmentationSample] = []

    for query_index, query_with_wordlist in enumerate(dataset.queries):
        if query_with_wordlist.query not in seen_texts:
            seen_texts.add(query_with_wordlist.query)
            samples.append(
                MoraSegmentationSample(
                    sample_id=f"q{query_index:02d}",
                    text=query_with_wordlist.query,
                    gold_moras=to_moras(query_with_wordlist.query),
                    source_type="query",
                    query_index=query_index,
                )
            )

        for candidate_index, candidate in enumerate(query_with_wordlist.wordlist):
            if candidate in seen_texts:
                continue
            seen_texts.add(candidate)
            samples.append(
                MoraSegmentationSample(
                    sample_id=f"q{query_index:02d}_c{candidate_index:03d}",
                    text=candidate,
                    gold_moras=to_moras(candidate),
                    source_type="candidate",
                    query_index=query_index,
                    candidate_index=candidate_index,
                )
            )
    return samples


def select_samples(
    samples: list[MoraSegmentationSample],
    *,
    sample_size: int,
    seed: int,
) -> list[MoraSegmentationSample]:
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if sample_size > len(samples):
        raise ValueError(
            f"sample_size must be <= total sample count ({len(samples)}), got {sample_size}"
        )
    selected = random.Random(seed).sample(samples, sample_size)
    return sorted(selected, key=lambda sample: (sample.query_index, sample.sample_id))


def build_system_prompt() -> str:
    return """
あなたはカタカナ文字列をモーラ列へ分割する判定器です。

- 入力はカタカナ文字列 1 つです
- 出力は、その文字列を左から順に分割したモーラ列です
- ン / ー / ッ はそれぞれ 1 モーラです
- ジャ や シュ のような拗音も 1 モーラです
- 説明や途中経過は出さず、JSON object だけを返してください
- 出力は {"moras": ["..."]} のみ
""".strip()


def build_user_prompt(sample: MoraSegmentationSample) -> str:
    return f"""
文字列: {sample.text}

モーラ列を JSON で返してください。
""".strip()


def infer_mora_segmentation(
    *,
    client: OpenAI,
    model_name: str,
    sample: MoraSegmentationSample,
    max_output_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": build_system_prompt(),
        "input": build_user_prompt(sample),
        "text_format": MoraSegmentationOutput,
        "max_output_tokens": max_output_tokens,
    }
    if model_name.startswith("gpt-5"):
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.parse(**request_kwargs)
    parsed_output = response.output_parsed
    if parsed_output is None:
        raise ValueError(f"Structured output was empty for {model_name}")

    predicted_moras = parsed_output.moras
    gold_moras = sample.gold_moras
    return {
        **asdict(sample),
        "predicted_moras": predicted_moras,
        "predicted_mora_count": len(predicted_moras),
        "gold_mora_count": len(gold_moras),
        "is_exact_match": predicted_moras == gold_moras,
        "mora_count_match": len(predicted_moras) == len(gold_moras),
        "response_id": response.id,
        "usage": usage_to_dict(response),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")

    exact_match_count = sum(1 for record in records if record["is_exact_match"])
    mora_count_match_count = sum(1 for record in records if record["mora_count_match"])
    gold_lengths = [record["gold_mora_count"] for record in records]
    predicted_lengths = [record["predicted_mora_count"] for record in records]
    absolute_count_errors = [
        abs(record["predicted_mora_count"] - record["gold_mora_count"]) for record in records
    ]

    return {
        "sample_count": len(records),
        "exact_match_rate": exact_match_count / len(records),
        "mora_count_match_rate": mora_count_match_count / len(records),
        "mean_absolute_mora_count_error": _mean(absolute_count_errors),
        "average_gold_mora_count": _mean(gold_lengths),
        "average_predicted_mora_count": _mean(predicted_lengths),
    }


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


def build_default_output_path(*, model_name: str, sample_size: int, seed: int) -> Path:
    model_name_safe = model_name.replace("/", "-")
    return (
        Path(__file__).resolve().parent
        / "results"
        / (
            "mora_segmentation_nonreasoning_"
            f"{model_name_safe}_small_sample{sample_size}_seed{seed}.json"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "small_dataset_for_llm 由来のカタカナ文字列を使って、LLM が reasoning "
            "なしでモーラ分割をどれだけ当てられるかを測る"
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
        help=f"評価する文字列数。デフォルト: {DEFAULT_SAMPLE_SIZE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"サンプリング用の seed。デフォルト: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--wordlist_size",
        type=int,
        default=100,
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
        default=128,
        help="Responses API の max_output_tokens。デフォルト: 128",
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
    all_samples = collect_unique_strings(wordlist_size=args.wordlist_size)
    sampled_strings = select_samples(
        all_samples,
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
        infer_mora_segmentation(
            client=client,
            model_name=args.model,
            sample=sample,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        for sample in sampled_strings
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
            "name": "small_dataset_for_llm_unique_strings",
            "total_sample_count": len(all_samples),
        },
        "metrics": {
            **compute_metrics(records),
            "usage": aggregate_usage(records),
        },
        "samples": records,
    }

    with output_file_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
