"""Source-aligned math validation followed by phonetic reranking transfer.

Every first model answer is saved before scoring. --replay is API-free.
"""

import argparse
import ast
import asyncio
from collections import Counter
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import random
import statistics
import sys
import time

import httpx
import numpy as np

COMMON = Path(__file__).parent / "common"
sys.path.insert(0, str(COMMON))
from greenblatt_prompts import build_few_shot_messages, build_user_message  # noqa: E402
from reranker import (  # noqa: E402
    _build_openai_json_schema_response_format,
    build_rerank_messages,
    get_prompt_config,
    get_rerank_response_format,
)

DATA = Path(__file__).parents[1] / "data/greenblatt_gen_arithmetic/cases.json"
CONDITIONS = {
    "baseline": (None, "numbers"),
    "counting_100": (100, "numbers"),
    "counting_300": (300, "numbers"),
    "ellipsis_300": (300, "ellipsis"),
}
SEED = 20261030


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def arithmetic_answer(problem):
    expression = problem.removeprefix("Evaluate this Python expression. ")
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (
                -evaluate(node.operand)
                if isinstance(node.op, ast.USub)
                else evaluate(node.operand)
            )
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("Unsupported arithmetic syntax")

    return evaluate(ast.parse(expression, mode="eval").body)


def math_data():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    shots, cases = data["few_shot"], data["cases"]
    if len(shots) != 10 or len(cases) != 600:
        raise ValueError("Expected 10 demonstrations and 600 cases")
    if {q["source_index"] for q in shots} & {q["source_index"] for q in cases}:
        raise ValueError("Demonstration leakage")
    if {q["problem"] for q in shots} & {q["problem"] for q in cases}:
        raise ValueError("Demonstration text leakage")
    for query in shots + cases:
        if arithmetic_answer(query["problem"]) != query["answer"]:
            raise ValueError("Incorrect arithmetic reference answer")
    return data


def math_messages(data, case, condition):
    length, filler_type = CONDITIONS[condition]
    pairs = [(row["source_index"], row) for row in data["few_shot"]]
    messages = build_few_shot_messages(
        pairs, filler_tokens=length, filler_type=filler_type, disable_prefill=True
    )
    result = [
        {
            "role": m["role"],
            "content": m["content"][0]["text"]
            if isinstance(m["content"], list)
            else m["content"],
        }
        for m in messages
    ]
    result.append(
        {
            "role": "user",
            "content": build_user_message(
                case, filler_tokens=length, filler_type=filler_type
            )
            + "\n\nAnswer:",
        }
    )
    return result


def phonetic_data():
    from soramimi_phonetic_search_dataset import load_default_dataset_for_llm
    from importlib.metadata import distribution

    source = json.loads(
        distribution("soramimi-phonetic-search-dataset").read_text("direct_url.json")
    )
    pin = "6072e13bed37dd2f8eb780e61b6154d21cff2e31"
    if source.get("vcs_info", {}).get("commit_id") != pin:
        raise ValueError("Pinned phonetic dataset required")
    dataset = load_default_dataset_for_llm(wordlist_size=100)
    if len(dataset.queries) != 150:
        raise ValueError("Expected 150 phonetic queries")
    return {
        "dataset_commit": pin,
        "cases": [
            {
                "source_index": i,
                "query": q.query,
                "candidate_words": q.wordlist,
                "positive_words": q.positive_words,
            }
            for i, q in enumerate(dataset.queries)
        ],
    }


def filler_text(condition):
    length, kind = CONDITIONS[condition]
    if length is None:
        return ""
    return (
        " ".join(map(str, range(1, length + 1)))
        if kind == "numbers"
        else " ".join(["..."] * length)
    )


def phonetic_messages(case, condition):
    config = get_prompt_config("detailed")
    filler = filler_text(condition)
    insertion = f"Filler: {filler}\n" if filler else ""
    # Demonstration and target both receive filler, as in the source math protocol.
    example = config.prompt_example_suffix.replace("Reranked:", insertion + "Reranked:")
    template = config.user_prompt_template.replace("Reranked:", insertion + "Reranked:")
    return build_rerank_messages(
        [case["query"]],
        [case["candidate_words"]],
        topn=10,
        prompt_template="detailed",
        prompt_example_suffix=example,
        user_prompt_template=template,
    )[0]


def build_manifest(model, task):
    data = math_data() if task == "math" else phonetic_data()
    requests = []
    for condition in CONDITIONS:
        for case in data["cases"]:
            body = {
                "model": model,
                "reasoning_effort": "none",
                "temperature": 0.0,
                "max_completion_tokens": 100 if task == "math" else 512,
                "messages": math_messages(data, case, condition)
                if task == "math"
                else phonetic_messages(case, condition),
            }
            if task == "phonetic":
                body["response_format"] = _build_openai_json_schema_response_format(
                    get_rerank_response_format(include_thoughts=False)
                )
            requests.append(
                {
                    "request_id": f"{condition}__q{case['source_index']:04d}",
                    "condition": condition,
                    "source_index": case["source_index"],
                    "body_sha256": digest(body),
                    "body": body,
                }
            )
    random.Random(SEED).shuffle(requests)
    return {
        "schema_version": 1,
        "task": task,
        "model": model,
        "data": data,
        "dataset_sha256": digest(data),
        "conditions": list(CONDITIONS),
        "primary_comparison": "counting_100 versus baseline",
        "schedule_seed": SEED,
        "replicates": 1,
        "requests": requests,
    }


def validate_record(record, request):
    if (
        record["request_id"] != request["request_id"]
        or record["body_sha256"] != request["body_sha256"]
    ):
        raise ValueError("Saved request mismatch")
    if (
        record["usage"].get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        != 0
    ):
        raise ValueError("Unexpected reasoning tokens")


async def execute_requests(manifest, output, concurrency=8, client=None):
    if client is None:
        async with httpx.AsyncClient(
            timeout=120,
            headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
        ) as opened:
            return await execute_requests(manifest, output, concurrency, opened)
    semaphore = asyncio.Semaphore(concurrency)
    completed, errors = 0, []
    started = time.monotonic()

    async def execute(request):
        nonlocal completed
        path = output / "responses" / (request["request_id"] + ".json")
        if path.exists():
            validate_record(json.loads(path.read_text()), request)
            completed += 1
            return
        async with semaphore:
            for attempt in range(4):
                try:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        json=request["body"],
                    )
                except httpx.TransportError:
                    if attempt < 3:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    errors.append(
                        {
                            "request_id": request["request_id"],
                            "error": "transport_error",
                        }
                    )
                    return
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < 3:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                if response.status_code != 200:
                    errors.append(
                        {
                            "request_id": request["request_id"],
                            "error": f"HTTP {response.status_code}",
                        }
                    )
                    return
                payload = response.json()
                # Save the successful provider response before any parsing or scoring.
                save(
                    output / "provider_responses" / (request["request_id"] + ".json"),
                    payload,
                )
                choice = payload["choices"][0]
                record = {
                    "request_id": request["request_id"],
                    "body_sha256": request["body_sha256"],
                    "model": payload["model"],
                    "content": choice["message"].get("content"),
                    "finish_reason": choice["finish_reason"],
                    "usage": payload["usage"],
                }
                save(path, record)
                validate_record(record, request)
                completed += 1
                if completed % 100 == 0:
                    print(
                        f"Saved {completed}/{len(manifest['requests'])}; {time.monotonic() - started:.0f}s",
                        flush=True,
                    )
                return

    await asyncio.gather(*(execute(request) for request in manifest["requests"]))
    save(output / "errors.json", errors)
    if errors:
        raise RuntimeError(
            f"{len(errors)} incomplete requests; successful answers are preserved"
        )


def score(record, case, task):
    if record["finish_reason"] != "stop" or not isinstance(record["content"], str):
        return {"score": 0.0, "valid": False, "prediction": None}
    text = record["content"].strip()
    try:
        if task == "math":
            # Match the source's numeric parser, but use None rather than a -1 sentinel.
            cleaned = (
                text.lower()
                .removeprefix("[")
                .strip()
                .removeprefix("answer:")
                .removesuffix("]")
                .strip()
            )
            prediction = int(cleaned.replace(",", ""))
            return {
                "score": float(prediction == case["answer"]),
                "valid": True,
                "prediction": prediction,
            }
        parsed = json.loads(text)
        prediction = parsed["reranked"]
        if (
            not isinstance(prediction, list)
            or len(prediction) != 10
            or any(type(i) is not int or not 0 <= i < 100 for i in prediction)
            or len(set(prediction)) != 10
        ):
            raise ValueError("Invalid ranking")
        positives = set(case["positive_words"])
        value = len({case["candidate_words"][i] for i in prediction} & positives) / len(
            positives
        )
        return {"score": value, "valid": True, "prediction": prediction}
    except (ValueError, KeyError, TypeError):
        return {"score": 0.0, "valid": False, "prediction": None}


def paired_summary(baseline, treatment, binary):
    delta = np.asarray(treatment) - np.asarray(baseline)
    rng = np.random.default_rng(SEED)
    draws = delta[rng.integers(0, len(delta), size=(20_000, len(delta)))].mean(axis=1)
    wins, losses = int((delta > 1e-12).sum()), int((delta < -1e-12).sum())
    result = {
        "delta": float(delta.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "improved": wins,
        "worsened": losses,
        "unchanged": len(delta) - wins - losses,
    }
    if binary:
        n = wins + losses
        result["mcnemar_exact_two_sided_p"] = (
            min(
                1.0,
                2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n,
            )
            if n
            else 1.0
        )
    return result


def summarize(manifest, output, records=None):
    if records is None:
        records = [
            json.loads((output / "responses" / (r["request_id"] + ".json")).read_text())
            for r in manifest["requests"]
        ]
    indexed = {r["request_id"]: r for r in records}
    if len(indexed) != len(records) or set(indexed) != {
        r["request_id"] for r in manifest["requests"]
    }:
        raise ValueError("Incomplete or duplicate response set")
    cases = {q["source_index"]: q for q in manifest["data"]["cases"]}
    scores = {c: {} for c in manifest["conditions"]}
    usage = {c: Counter() for c in manifest["conditions"]}
    models = set()
    for request in manifest["requests"]:
        record = indexed[request["request_id"]]
        validate_record(record, request)
        condition, index = request["condition"], request["source_index"]
        scores[condition][index] = score(record, cases[index], manifest["task"])
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            usage[condition][key] += record["usage"][key]
        usage[condition]["cached_input_tokens"] += (
            record["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        )
        models.add(record["model"])
    order = sorted(cases)
    cells = {}
    for condition in manifest["conditions"]:
        values = [scores[condition][i]["score"] for i in order]
        cells[condition] = {
            "mean": statistics.mean(values),
            "n": len(values),
            "invalid_outputs": sum(
                not row["valid"] for row in scores[condition].values()
            ),
            "scores": values,
            "usage": dict(usage[condition]),
        }
        if condition != "baseline":
            cells[condition]["versus_baseline"] = paired_summary(
                [scores["baseline"][i]["score"] for i in order],
                values,
                manifest["task"] == "math",
            )
    result = {
        "task": manifest["task"],
        "metric": "accuracy" if manifest["task"] == "math" else "macro Recall@10",
        "model": manifest["model"],
        "models_returned": sorted(models),
        "dataset_sha256": manifest["dataset_sha256"],
        "primary_comparison": manifest["primary_comparison"],
        "query_order": order,
        "cells": cells,
        "ci_method": "Paired case bootstrap; 20000 draws; seed 20261030; unadjusted exploratory intervals except predeclared primary comparison",
    }
    save(output / "summary.json", result)
    save(
        output / "results.json",
        {
            "protocol": {k: v for k, v in manifest.items() if k != "requests"},
            "responses": sorted(records, key=lambda r: r["request_id"]),
        },
    )
    for c, cell in cells.items():
        print(c, f"{cell['mean']:.6f}", cell.get("versus_baseline", ""), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=["math", "phonetic"], default="math")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--summarize-only", action="store_true")
    mode.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    manifest = build_manifest(args.model, args.task)
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a new output directory")
    if not path.exists():
        save(path, manifest)
    print(f"Prepared {len(manifest['requests'])} requests", flush=True)
    if args.prepare_only:
        return
    if args.replay:
        exported = json.loads(args.replay.read_text())
        if exported["protocol"] != {
            k: v for k, v in manifest.items() if k != "requests"
        }:
            raise ValueError("Replay protocol mismatch")
        summarize(manifest, args.output, exported["responses"])
    else:
        if not args.summarize_only:
            asyncio.run(execute_requests(manifest, args.output, args.concurrency))
        summarize(manifest, args.output)


if __name__ == "__main__":
    main()
