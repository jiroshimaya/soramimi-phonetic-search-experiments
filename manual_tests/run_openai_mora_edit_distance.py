"""
OpenAI API を使って、LLM が 3 件のモーラ編集距離を正しく返すかを手で確認する。

実行前に `source ~/.zshrc` で API キーを読み込む前提。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from openai import OpenAI

from analytics.mora_edit_distance_nonreasoning import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT_MODE,
    PROMPT_MODES,
    build_mora_distance_pair,
    infer_distance,
)


@dataclass(frozen=True)
class ManualTestCase:
    case_id: str
    query: str
    candidate: str
    expected_distance: int


MANUAL_TEST_CASES = [
    ManualTestCase(
        case_id="identical",
        query="アイ",
        candidate="アイ",
        expected_distance=0,
    ),
    ManualTestCase(
        case_id="single-substitution",
        query="アイ",
        candidate="アオ",
        expected_distance=1,
    ),
    ManualTestCase(
        case_id="single-deletion",
        query="アウマル",
        candidate="アウマ",
        expected_distance=1,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI API で 3 件のモーラ編集距離 manual test を実行する"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"使用するモデル名。デフォルト: {DEFAULT_MODEL}",
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
        default=32,
        help="Responses API の max_output_tokens。デフォルト: 32",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default=DEFAULT_PROMPT_MODE,
        choices=PROMPT_MODES,
        help="入力の与え方。mora_spaced または kana_only。デフォルト: mora_spaced",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI()
    results = []

    for test_case in MANUAL_TEST_CASES:
        pair = build_mora_distance_pair(
            pair_id=test_case.case_id,
            query=test_case.query,
            candidate=test_case.candidate,
        )
        result = infer_distance(
            client=client,
            model_name=args.model,
            pair=pair,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            prompt_mode=args.prompt_mode,
        )
        expected_distance = test_case.expected_distance
        predicted_distance = result["predicted_distance"]
        if predicted_distance != expected_distance:
            raise AssertionError(
                f"{test_case.case_id}: expected {expected_distance}, got {predicted_distance}"
            )
        results.append(
            {
                "case_id": test_case.case_id,
                "query": test_case.query,
                "candidate": test_case.candidate,
                "expected_distance": expected_distance,
                "predicted_distance": predicted_distance,
                "prompt_mode": args.prompt_mode,
                "response_id": result["response_id"],
                "usage": result["usage"],
            }
        )

    print(
        json.dumps(
            {
                "model": args.model,
                "prompt_mode": args.prompt_mode,
                "case_count": len(results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
