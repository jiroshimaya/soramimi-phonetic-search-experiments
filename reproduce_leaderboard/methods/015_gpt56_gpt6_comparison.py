"""Prepare an offline, comparable GPT-5.6/GPT-6 evaluation manifest.

Usage: uv run reproduce_leaderboard/methods/015_gpt56_gpt6_comparison.py --output PATH
This command never submits requests or verifies free-quota eligibility.
"""

import argparse
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "common"))

from reranker import (  # noqa: E402
    _build_openai_json_schema_response_format,
    build_rerank_messages,
    get_prompt_config,
    get_rerank_response_format,
)
from soramimi_phonetic_search_dataset import load_default_dataset_for_llm  # noqa: E402

DATASET_COMMIT = "6072e13bed37dd2f8eb780e61b6154d21cff2e31"
MODELS = ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra")
QUERY_COUNT = 150
WORDLIST_SIZE = 100
TOPN = 10
MAX_COMPLETION_TOKENS = 24_000
FRAMING_RESERVATION = 256


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def dataset_provenance() -> dict:
    package = distribution("soramimi-phonetic-search-dataset")
    source = json.loads(package.read_text("direct_url.json") or "{}")
    commit = source.get("vcs_info", {}).get("commit_id")
    if commit != DATASET_COMMIT:
        raise ValueError(f"Dataset must be installed at pinned commit {DATASET_COMMIT}")
    return {
        "package": package.metadata["Name"],
        "version": package.version,
        "commit": commit,
        "loader": "load_default_dataset_for_llm(wordlist_size=100)",
    }


def build_manifest() -> dict:
    provenance = dataset_provenance()
    dataset = load_default_dataset_for_llm(wordlist_size=WORDLIST_SIZE)
    if len(dataset.queries) != QUERY_COUNT:
        raise ValueError(f"Expected {QUERY_COUNT} queries")
    queries = []
    for index, query in enumerate(dataset.queries):
        if len(query.wordlist) != WORDLIST_SIZE:
            raise ValueError(f"Query {index} must contain {WORDLIST_SIZE} candidates")
        if not query.positive_words or not set(query.positive_words).issubset(
            query.wordlist
        ):
            raise ValueError(f"Query {index} has missing positive candidates")
        query_data = {
            "query_index": index,
            "query": query.query,
            "candidate_words": list(query.wordlist),
            "positive_words": list(query.positive_words),
        }
        queries.append({**query_data, "sha256": digest(query_data)})

    config = get_prompt_config("step_by_step")
    prompt = {
        "template": "step_by_step",
        "variant": "v1",
        "source": "reproduce_leaderboard/methods/common/reranker.py",
        "baseline": "010_03_llm_rerank_gpt54_medium_step_by_step",
        "instructions": config.prompt_instructions.strip(),
        "example_suffix": config.prompt_example_suffix.strip(),
        "user_template": config.user_prompt_template,
        "input_transform": "none",
    }
    messages = build_rerank_messages(
        [query["query"] for query in queries],
        [query["candidate_words"] for query in queries],
        topn=TOPN,
        prompt_template="step_by_step",
        input_transform="none",
    )
    response_format = _build_openai_json_schema_response_format(
        get_rerank_response_format(include_thoughts=False)
    )
    requests = []
    for model in MODELS:
        for query, query_messages in zip(queries, messages, strict=True):
            body = {
                "model": model,
                "messages": query_messages,
                "reasoning_effort": "medium",
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "response_format": response_format,
            }
            input_reservation = len(canonical_bytes(body)) + FRAMING_RESERVATION
            requests.append(
                {
                    "request_id": f"{model}-medium-q{query['query_index']:04d}",
                    "query_index": query["query_index"],
                    "query_sha256": query["sha256"],
                    "body": body,
                    "body_sha256": digest(body),
                    "token_reservation": {
                        "input": input_reservation,
                        "completion": MAX_COMPLETION_TOKENS,
                        "total": input_reservation + MAX_COMPLETION_TOKENS,
                    },
                }
            )
    return {
        "schema_version": 1,
        "status": "offline_prepared_not_submitted",
        "free_quota_eligibility": "unverified",
        "api_parameter_compatibility": "unverified",
        "dataset": {**provenance, "sha256": digest(queries)},
        "prompt": {**prompt, "sha256": digest(prompt)},
        "evaluation": {
            "metric": "macro Recall@10",
            "topn": TOPN,
            "reasoning_effort": "medium",
            "models": list(MODELS),
            "query_count": len(queries),
            "candidates_per_query": WORDLIST_SIZE,
            "job_count": len(MODELS),
            "request_count": len(requests),
        },
        "token_reservation": {
            "method": "UTF-8 bytes of canonical request JSON (including schema) + 256 framing tokens + max_completion_tokens",
            "note": "Conservative planning reservation, not predicted usage or a verified provider token bound; does not establish free-quota eligibility.",
            "total": sum(item["token_reservation"]["total"] for item in requests),
        },
        "queries": queries,
        "requests": requests,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    evaluation = manifest["evaluation"]
    print(
        f"Prepared {evaluation['query_count']} queries x {evaluation['job_count']} "
        f"models = {evaluation['request_count']} requests: {args.output}"
    )
    print("Offline only; free-quota eligibility and API compatibility are unverified.")


if __name__ == "__main__":
    main()
