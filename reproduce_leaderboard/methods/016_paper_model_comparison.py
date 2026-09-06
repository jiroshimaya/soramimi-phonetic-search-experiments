"""Prepare the full prompt/reasoning comparison without submitting API requests."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

METHODS = Path(__file__).parent
sys.path.insert(0, str(METHODS / "common"))

from prompt_variations import PROMPT_VARIATIONS  # noqa: E402
from reranker import build_rerank_messages, get_prompt_config  # noqa: E402

MODELS = ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra")
EFFORTS = ("none", "medium")
PROMPTS = ("simple", "detailed", "step_by_step")
VARIANTS = ("v1", "v2", "v3", "v4", "v5")
SUBSETS_PATH = METHODS.parent / "data/paper_subsets.json"


def base_protocol():
    spec = importlib.util.spec_from_file_location(
        "comparison_015", METHODS / "015_gpt56_gpt6_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_subsets(queries):
    subsets = json.loads(SUBSETS_PATH.read_text(encoding="utf-8"))
    mapping = subsets["query_to_subset"]
    if set(mapping) != {query["query"] for query in queries}:
        raise ValueError("Frozen difficulty mapping does not match the query set")
    if Counter(mapping[query["query"]] for query in queries) != {
        "easy": 65,
        "medium": 47,
        "hard": 38,
    }:
        raise ValueError("Unexpected difficulty counts")
    return subsets


def job_id(model, effort, prompt, variant):
    return f"{model}__{effort}__{prompt}__{variant}"


def build_manifest():
    protocol = base_protocol()
    original = protocol.build_manifest()
    queries = original["queries"]
    subsets = load_subsets(queries)
    response_format = original["requests"][0]["body"]["response_format"]
    conditions = [("medium", "step_by_step", variant) for variant in VARIANTS]
    conditions.extend(
        (effort, prompt, variant)
        for effort in EFFORTS
        for prompt in PROMPTS
        for variant in VARIANTS
        if (effort, prompt) != ("medium", "step_by_step")
    )
    jobs, requests = [], []
    for effort, prompt, variant in conditions:
        config = get_prompt_config(prompt)
        instructions = (
            config.prompt_instructions
            if variant == "v1"
            else PROMPT_VARIATIONS[prompt][variant]
        )
        prompt_fields = {
            "prompt_instructions": instructions.strip(),
            "prompt_example_suffix": config.prompt_example_suffix.strip(),
            "user_prompt_template": config.user_prompt_template,
        }
        prompt_hash = protocol.digest(prompt_fields)
        for model in MODELS:
            jobs.append(
                {
                    "job_id": job_id(model, effort, prompt, variant),
                    "model": model,
                    "reasoning_effort": effort,
                    "prompt_template": prompt,
                    "prompt_variant": variant,
                    "prompt_sha256": prompt_hash,
                    **prompt_fields,
                }
            )
        messages = build_rerank_messages(
            [query["query"] for query in queries],
            [query["candidate_words"] for query in queries],
            topn=10,
            prompt_template=prompt,
            prompt_instructions=instructions,
            prompt_example_suffix=config.prompt_example_suffix,
            user_prompt_template=config.user_prompt_template,
            input_transform="none",
        )
        for query, query_messages in zip(queries, messages, strict=True):
            for model in MODELS:
                identifier = job_id(model, effort, prompt, variant)
                body = {
                    "model": model,
                    "messages": query_messages,
                    "reasoning_effort": effort,
                    "max_completion_tokens": 1000 if effort == "none" else 24_000,
                    "response_format": response_format,
                }
                input_reservation = len(protocol.canonical_bytes(body)) + 1024
                completion_reservation = body["max_completion_tokens"]
                requests.append(
                    {
                        "request_id": f"{identifier}__q{query['query_index']:04d}",
                        "job_id": identifier,
                        "query_index": query["query_index"],
                        "query_sha256": query["sha256"],
                        "body": body,
                        "body_sha256": protocol.digest(body),
                        "token_reservation": {
                            "input": input_reservation,
                            "completion": completion_reservation,
                            "total": input_reservation + completion_reservation,
                        },
                    }
                )
    return {
        "schema_version": 2,
        "status": "offline_prepared_not_submitted",
        "free_quota_eligibility": "requires_current_account_verification",
        "unavailable_models": [
            {"model": "gpt-6-astra", "reason": "free_eligibility_unconfirmed"}
        ],
        "dataset": original["dataset"],
        "queries": queries,
        "subsets": subsets,
        "jobs": jobs,
        "evaluation": {
            "metric": "macro Recall@10",
            "topn": 10,
            "models": list(MODELS),
            "reasoning_efforts": list(EFFORTS),
            "prompt_templates": list(PROMPTS),
            "prompt_variants": list(VARIANTS),
            "query_count": len(queries),
            "candidates_per_query": 100,
            "job_count": len(jobs),
            "request_count": len(requests),
            "input_transform": "none",
            "aggregate": "mean and sample standard deviation over five complete variants",
            "standard_deviation_ddof": 1,
        },
        "token_reservation": {
            "method": "Canonical request UTF-8 bytes + 1024 framing + max_completion_tokens",
            "note": "Conservative planning reservation, not predicted usage or a verified provider token bound.",
            "total": sum(request["token_reservation"]["total"] for request in requests),
        },
        "requests": requests,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(
        f"Prepared {len(manifest['jobs'])} jobs / {len(manifest['requests'])} requests: {args.output}"
    )
    print(
        "Offline only; daily account eligibility and remaining quota require verification."
    )


if __name__ == "__main__":
    main()
