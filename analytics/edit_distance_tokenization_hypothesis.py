"""
編集距離タスクの exact match と、入力文字列のトークン分割の対応関係を調べる補助スクリプト。

`gpt-5.4` 用の tokenizer は `tiktoken` で直接解決できないため、
近い OpenAI 系モデルで使われている `o200k_base` を proxy として使う。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken


ENCODING_NAME = "o200k_base"
DEFAULT_RESULT_FILES = [
    "analytics/results/mora_edit_distance_nonreasoning_gpt-5.4_small_sample100_seed7.json",
    "analytics/results/kana_edit_distance_nonreasoning_gpt-5.4_small_sample100_seed7.json",
    "analytics/results/kana_edit_distance_nonreasoning_gpt-5.4_promptkana_spaced_small_sample100_seed7.json",
]


@dataclass(frozen=True)
class AlignmentStats:
    strict_unit_token_match: bool
    boundary_aligned: bool
    token_count: int
    unit_count: int


def token_bytes(encoding: tiktoken.Encoding, text: str) -> list[bytes]:
    return [encoding.decode_single_token_bytes(token) for token in encoding.encode(text)]


def build_segments(units: list[str], *, spaced: bool) -> list[str]:
    if not units:
        return []
    if not spaced:
        return units
    return [units[0], *[f" {unit}" for unit in units[1:]]]


def cumulative_lengths(byte_pieces: list[bytes]) -> list[int]:
    total = 0
    result: list[int] = []
    for piece in byte_pieces:
        total += len(piece)
        result.append(total)
    return result


def analyze_alignment(
    encoding: tiktoken.Encoding,
    *,
    rendered_text: str,
    units: list[str],
    spaced: bool,
) -> AlignmentStats:
    token_pieces = token_bytes(encoding, rendered_text)
    segment_pieces = [segment.encode("utf-8") for segment in build_segments(units, spaced=spaced)]
    return AlignmentStats(
        strict_unit_token_match=token_pieces == segment_pieces,
        boundary_aligned=cumulative_lengths(token_pieces) == cumulative_lengths(segment_pieces),
        token_count=len(token_pieces),
        unit_count=len(segment_pieces),
    )


def mode_config(record: dict[str, Any]) -> tuple[str, bool]:
    prompt_mode = record.get("prompt_mode")
    if prompt_mode is None:
        prompt_mode = "mora_spaced" if "query_moras" in record else "kana_only"
    if prompt_mode == "mora_spaced":
        return "mora", True
    if prompt_mode == "kana_spaced":
        return "kana", True
    if prompt_mode == "kana_only":
        return "kana", False
    if prompt_mode == "kana_only_with_mora_hint":
        return "kana", False
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def units_for_text(record: dict[str, Any], field: str, unit_kind: str) -> list[str]:
    text = record[field]
    if unit_kind == "kana":
        return list(text)
    if unit_kind == "mora":
        return record[f"{field}_moras"]
    raise ValueError(f"Unsupported unit_kind: {unit_kind}")


def rendered_text(record: dict[str, Any], field: str, *, spaced: bool) -> str:
    text = record[field]
    return " ".join(list(text)) if spaced and field in {"query", "candidate"} and "query_moras" not in record else text


def rendered_text_for_units(record: dict[str, Any], field: str, unit_kind: str, spaced: bool) -> str:
    if unit_kind == "mora":
        units = record[f"{field}_moras"]
        return " ".join(units) if spaced else "".join(units)
    text = record[field]
    return " ".join(list(text)) if spaced else text


def analyze_record(encoding: tiktoken.Encoding, record: dict[str, Any]) -> dict[str, Any]:
    unit_kind, spaced = mode_config(record)
    query_units = units_for_text(record, "query", unit_kind)
    candidate_units = units_for_text(record, "candidate", unit_kind)
    query_alignment = analyze_alignment(
        encoding,
        rendered_text=rendered_text_for_units(record, "query", unit_kind, spaced),
        units=query_units,
        spaced=spaced,
    )
    candidate_alignment = analyze_alignment(
        encoding,
        rendered_text=rendered_text_for_units(record, "candidate", unit_kind, spaced),
        units=candidate_units,
        spaced=spaced,
    )
    return {
        **record,
        "alignment": {
            "unit_kind": unit_kind,
            "spaced": spaced,
            "query": query_alignment.__dict__,
            "candidate": candidate_alignment.__dict__,
            "pair_strict_unit_token_match": (
                query_alignment.strict_unit_token_match
                and candidate_alignment.strict_unit_token_match
            ),
            "pair_boundary_aligned": (
                query_alignment.boundary_aligned and candidate_alignment.boundary_aligned
            ),
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    pair_count = len(records)
    strict_match_pairs = [
        record for record in records if record["alignment"]["pair_strict_unit_token_match"]
    ]
    misaligned_pairs = [
        record for record in records if not record["alignment"]["pair_strict_unit_token_match"]
    ]
    exact_match_pairs = [record for record in records if record["is_exact_match"]]
    exact_match_on_misaligned_pairs = [
        record for record in misaligned_pairs if record["is_exact_match"]
    ]
    non_exact_on_strict_match_pairs = [
        record for record in strict_match_pairs if not record["is_exact_match"]
    ]
    boundary_aligned_pairs = [
        record for record in records if record["alignment"]["pair_boundary_aligned"]
    ]
    return {
        "pair_count": pair_count,
        "strict_unit_token_match_pair_count": len(strict_match_pairs),
        "strict_unit_token_match_rate": len(strict_match_pairs) / pair_count,
        "boundary_aligned_pair_count": len(boundary_aligned_pairs),
        "boundary_aligned_pair_rate": len(boundary_aligned_pairs) / pair_count,
        "exact_match_rate": len(exact_match_pairs) / pair_count,
        "exact_match_on_misaligned_pair_count": len(exact_match_on_misaligned_pairs),
        "exact_match_on_misaligned_pair_rate": len(exact_match_on_misaligned_pairs)
        / pair_count,
        "non_exact_on_strict_match_pair_count": len(non_exact_on_strict_match_pairs),
        "hypothesis_all_misaligned_fail_exact_match": len(exact_match_on_misaligned_pairs)
        == 0,
        "hypothesis_all_strict_match_succeed": len(non_exact_on_strict_match_pairs) == 0,
        "example_exact_match_on_misaligned_pairs": [
            {
                "pair_id": record["pair_id"],
                "query": record["query"],
                "candidate": record["candidate"],
            }
            for record in exact_match_on_misaligned_pairs[:5]
        ],
    }


def analyze_result_file(path: Path, encoding: tiktoken.Encoding) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = [analyze_record(encoding, record) for record in data["pairs"]]
    return {
        "path": str(path),
        "encoding": ENCODING_NAME,
        "parameters": data["parameters"],
        "summary": summarize(records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="編集距離結果 JSON に対して token/unit alignment 仮説を検証する"
    )
    parser.add_argument(
        "result_files",
        nargs="*",
        default=DEFAULT_RESULT_FILES,
        help="分析対象の result JSON。省略時は既定の3ファイルを使う",
    )
    parser.add_argument("-o", "--output_file_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    analysis = [
        analyze_result_file(Path(result_file), encoding)
        for result_file in args.result_files
    ]
    output = {"encoding_proxy": ENCODING_NAME, "analyses": analysis}

    if args.output_file_path is not None:
        args.output_file_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_file_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
