"""Paired filler-token experiment on the pinned 150-query reranking task.

Run with --output DIR; rerunning resumes identical saved requests.
No API calls are made by --prepare-only, --summarize-only, or --replay.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

import httpx
import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).parent / "common"))
from reranker import (
    build_rerank_messages,
    get_prompt_config,
    get_rerank_response_format,
    _build_openai_json_schema_response_format,
)  # noqa: E402
from soramimi_phonetic_search_dataset import load_default_dataset_for_llm  # noqa: E402

PIN = "6072e13bed37dd2f8eb780e61b6154d21cff2e31"
MODEL = "gpt-5.4"
FILLERS = {
    "baseline": "",
    "dots_300_chars": "." * 300,
    "dots_300_tokens": " ".join(["."] * 300),
}
SOURCE = "https://joisino.hatenablog.com/entry/filler"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_manifest(repeats=3, structured=False, counting=False):
    fillers = (
        {"baseline": "", "counting_300": " ".join(map(str, range(1, 301)))}
        if counting
        else FILLERS
    )
    if repeats < 1:
        raise ValueError("repeats must be positive")
    direct = json.loads(
        distribution("soramimi-phonetic-search-dataset").read_text("direct_url.json")
    )
    if direct.get("vcs_info", {}).get("commit_id") != PIN:
        raise ValueError("Pinned dataset required")
    dataset = load_default_dataset_for_llm(wordlist_size=100)
    if len(dataset.queries) != 150:
        raise ValueError("Expected 150 queries")
    queries = []
    for index, query in enumerate(dataset.queries):
        if (
            len(query.wordlist) != 100
            or not query.positive_words
            or not set(query.positive_words).issubset(query.wordlist)
        ):
            raise ValueError("Invalid candidates or positive words")
        queries.append(
            {
                "query_index": index,
                "query": query.query,
                "candidate_words": query.wordlist,
                "positive_words": query.positive_words,
            }
        )
    encoding = tiktoken.get_encoding("o200k_base")
    config = get_prompt_config("detailed")
    requests = []
    for condition, filler in fillers.items():
        # Keep the answer cue after the filler; preserve every other prompt byte.
        template = config.user_prompt_template.replace(
            "Reranked:", filler + ("\n" if filler else "") + "Reranked:"
        )
        messages = build_rerank_messages(
            [q["query"] for q in queries],
            [q["candidate_words"] for q in queries],
            topn=10,
            prompt_template="detailed",
            user_prompt_template=template,
        )
        for repeat in range(repeats):
            for query, message in zip(queries, messages, strict=True):
                body = {
                    "model": MODEL,
                    "reasoning_effort": "none",
                    "max_completion_tokens": 512,
                    "messages": message,
                }
                if structured:
                    body["response_format"] = _build_openai_json_schema_response_format(
                        get_rerank_response_format(include_thoughts=False)
                    )
                requests.append(
                    {
                        "request_id": f"{condition}__r{repeat}__q{query['query_index']:03d}",
                        "condition": condition,
                        "repeat": repeat,
                        "query_index": query["query_index"],
                        "body": body,
                        "body_sha256": digest(body),
                    }
                )
    random.Random(20261027).shuffle(requests)
    return {
        "dataset_commit": PIN,
        "dataset_sha256": digest(queries),
        "queries": queries,
        "source": SOURCE,
        "model": MODEL,
        "prompt_template": "detailed",
        "reasoning_effort": "none",
        "output_format": "json_schema" if structured else "plain_indices",
        "repeats": repeats,
        "topn": 10,
        "invalid_output_policy": "Save first model response and score invalid or incomplete rankings as zero; do not retry model output",
        "tokenizer": "o200k_base",
        "fillers": {
            name: {"text": s, "characters": len(s), "tokens": len(encoding.encode(s))}
            for name, s in fillers.items()
        },
        "requests": requests,
        "schedule_seed": 20261027,
    }


def parse_indices(content, candidate_count=100):
    if isinstance(content, str) and (
        content.strip().startswith("[") != content.strip().endswith("]")
    ):
        raise ValueError("Unbalanced brackets")
    if not isinstance(content, str) or not re.fullmatch(
        r"\s*\[?\s*\d+(?:\s*,\s*\d+)*\s*\]?\s*", content
    ):
        raise ValueError("Response must contain only comma-separated indices")
    indices = [int(x) for x in re.findall(r"\d+", content)]
    if (
        len(indices) != 10
        or len(set(indices)) != 10
        or any(x < 0 or x >= candidate_count for x in indices)
    ):
        raise ValueError("Expected ten distinct candidate indices in range")
    return indices


def score(indices, query):
    positives = set(query["positive_words"])
    return len({query["candidate_words"][i] for i in indices} & positives) / len(
        positives
    )


def parse_answer(content, finish_reason, structured=False):
    if finish_reason != "stop":
        raise ValueError("Incomplete response")
    if structured:
        payload = json.loads(content)
        if not isinstance(payload, dict) or set(payload) != {"reranked"}:
            raise ValueError("Unexpected structured output")
        indices = payload["reranked"]
        if not isinstance(indices, list) or any(type(i) is not int for i in indices):
            raise ValueError("Expected integer indices")
        return parse_indices(",".join(map(str, indices)))
    return parse_indices(content)


def validate_record(record, request):
    if (
        record["request_id"] != request["request_id"]
        or record["body_sha256"] != request["body_sha256"]
    ):
        raise ValueError("Saved response request mismatch")
    if (
        record.get("usage", {})
        .get("completion_tokens_details", {})
        .get("reasoning_tokens", 0)
        != 0
    ):
        raise ValueError("Unexpected reasoning tokens")
    try:
        indices = parse_answer(
            record["content"],
            record["finish_reason"],
            "response_format" in request.get("body", {}),
        )
    except ValueError as exc:
        if (
            record.get("valid_ranking") is not False
            or record["indices"] != []
            or record.get("parse_error") != str(exc)
        ):
            raise ValueError("Saved invalid response mismatch") from exc
        return []
    if indices != record["indices"] or record.get("valid_ranking") is not True:
        raise ValueError("Saved indices mismatch")
    return indices


async def run(manifest, output, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    failures = []
    started = time.monotonic()
    async with httpx.AsyncClient(
        timeout=120, headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    ) as client:

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
                        if response.status_code == 429 or response.status_code >= 500:
                            if attempt < 3:
                                await asyncio.sleep(min(30, 2 ** (attempt + 1)))
                                continue
                        if response.status_code != 200:
                            raise RuntimeError(f"HTTP {response.status_code}")
                        data = response.json()
                        choice = data["choices"][0]
                        content = choice["message"]["content"]
                        try:
                            indices = parse_answer(
                                content,
                                choice["finish_reason"],
                                "response_format" in request["body"],
                            )
                            valid_ranking, parse_error = True, None
                        except ValueError as exc:
                            indices, valid_ranking, parse_error = [], False, str(exc)
                        usage = data["usage"]
                        if (
                            usage.get("completion_tokens_details", {}).get(
                                "reasoning_tokens", 0
                            )
                            != 0
                        ):
                            raise ValueError("Unexpected reasoning tokens")
                        save(
                            path,
                            {
                                "request_id": request["request_id"],
                                "body_sha256": request["body_sha256"],
                                "model": data["model"],
                                "content": content,
                                "indices": indices,
                                "valid_ranking": valid_ranking,
                                "parse_error": parse_error,
                                "usage": usage,
                                "finish_reason": choice["finish_reason"],
                                "created_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                        completed += 1
                        if completed % 50 == 0:
                            print(
                                f"Completed {completed}/{len(manifest['requests'])}; elapsed {time.monotonic() - started:.0f}s",
                                flush=True,
                            )
                        return
                    except httpx.TransportError:
                        if attempt < 3:
                            await asyncio.sleep(2 ** (attempt + 1))
                            continue
                        failures.append(
                            {
                                "request_id": request["request_id"],
                                "error": "transport_error",
                            }
                        )
                        return
                    except (ValueError, KeyError, RuntimeError) as exc:
                        # Never retry malformed model answers: no success-selection bias.
                        failures.append(
                            {"request_id": request["request_id"], "error": str(exc)}
                        )
                        return

        await asyncio.gather(*(execute(request) for request in manifest["requests"]))
    save(output / "failures.json", failures)
    if failures:
        raise RuntimeError(
            f"{len(failures)} failed requests; inspect failures.json before resuming"
        )


def paired_stats(baseline, treatment):
    baseline = np.asarray(baseline, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if baseline.shape != treatment.shape or baseline.ndim != 2:
        raise ValueError("Expected matched repeat-by-query arrays")
    # Resample queries, keeping all repeats together (queries are the unit).
    delta = (treatment - baseline).mean(axis=0)
    rng = np.random.default_rng(20261027)
    draws = delta[rng.integers(0, len(delta), size=(20_000, len(delta)))].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "improved_queries": int((delta > 1e-12).sum()),
        "worsened_queries": int((delta < -1e-12).sum()),
        "unchanged_queries": int((np.abs(delta) <= 1e-12).sum()),
    }


def summarize(manifest, output):
    scores = {
        c: [[None] * 150 for _ in range(manifest["repeats"])]
        for c in manifest["fillers"]
    }
    cells = {
        c: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "invalid_outputs": 0,
        }
        for c in manifest["fillers"]
    }
    models = set()
    records = []
    for request in manifest["requests"]:
        path = output / "responses" / (request["request_id"] + ".json")
        record = json.loads(path.read_text())
        indices = validate_record(record, request)
        records.append(
            {key: value for key, value in record.items() if key != "created_at"}
        )
        condition, repeat, index = (
            request["condition"],
            request["repeat"],
            request["query_index"],
        )
        scores[condition][repeat][index] = score(indices, manifest["queries"][index])
        usage = record["usage"]
        cells[condition]["invalid_outputs"] += int(not record["valid_ranking"])
        cells[condition]["input_tokens"] += usage["prompt_tokens"]
        cells[condition]["output_tokens"] += usage["completion_tokens"]
        cells[condition]["cached_input_tokens"] += usage.get(
            "prompt_tokens_details", {}
        ).get("cached_tokens", 0)
        models.add(record["model"])
    for condition in manifest["fillers"]:
        if any(value is None for repeat in scores[condition] for value in repeat):
            raise ValueError("Incomplete condition")
        means = [statistics.mean(repeat) for repeat in scores[condition]]
        cells[condition].update(
            {
                "recall_at_10": statistics.mean(means),
                "repeat_recalls": means,
                "repeat_stdev": statistics.stdev(means) if len(means) > 1 else None,
                "query_recalls_by_repeat": scores[condition],
            }
        )
        if condition != "baseline":
            cells[condition]["versus_baseline"] = paired_stats(
                scores["baseline"], scores[condition]
            )
    summary = {
        "metric": "macro Recall@10",
        "query_count": 150,
        "candidates_per_query": 100,
        "candidate_protocol": "All positive words plus hard negatives; conditional reranking, not end-to-end retrieval",
        "dataset_commit": PIN,
        "dataset_sha256": manifest["dataset_sha256"],
        "models_returned": sorted(models),
        "prompt_template": manifest["prompt_template"],
        "output_format": manifest["output_format"],
        "reasoning_effort": "none",
        "repeats": manifest["repeats"],
        "request_count": len(manifest["requests"]),
        "fillers": manifest["fillers"],
        "ci_method": "Paired query bootstrap, repeats averaged within query; 20000 draws, seed 20261027; unadjusted exploratory 95% intervals",
        "cells": cells,
    }
    save(output / "summary.json", summary)
    config = get_prompt_config(manifest["prompt_template"])
    save(
        output / "results.json",
        {
            "protocol": {
                key: value for key, value in manifest.items() if key != "requests"
            },
            "prompt_config": {
                "instructions": config.prompt_instructions,
                "example_suffix": config.prompt_example_suffix,
                "user_template": config.user_prompt_template,
            },
            "responses": sorted(records, key=lambda record: record["request_id"]),
        },
    )
    for condition, cell in cells.items():
        print(
            condition,
            f"Recall@10={cell['recall_at_10']:.6f}",
            cell.get("versus_baseline", ""),
        )
    return summary


def reuse_records(manifest, output, source):
    """Reuse matching replicate slots, never collapse independent repetitions."""
    reused = 0
    for request in manifest["requests"]:
        name = request["request_id"] + ".json"
        previous = source / "responses" / name
        destination = output / "responses" / name
        if destination.exists() or not previous.exists():
            continue
        record = json.loads(previous.read_text(encoding="utf-8"))
        # Matching names alone are insufficient: prompts/model/settings must match.
        if record["body_sha256"] != request["body_sha256"]:
            continue
        validate_record(record, request)
        save(destination, record)
        reused += 1
    return reused


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Use the existing benchmark JSON schema",
    )
    parser.add_argument(
        "--counting",
        action="store_true",
        help="Compare counting from 1 to 300 against baseline, following Greenblatt",
    )
    parser.add_argument(
        "--reuse-output",
        type=Path,
        help="Reuse matching saved request/replicate slots from this output directory",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--summarize-only", action="store_true")
    mode.add_argument(
        "--replay",
        type=Path,
        help="Recompute from an exported results.json without API calls",
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "responses").mkdir(exist_ok=True)
    manifest = build_manifest(
        args.repeats, structured=args.structured, counting=args.counting
    )
    path = args.output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Existing manifest differs; use another output directory")
    else:
        save(path, manifest)
    print(f"Prepared {len(manifest['requests'])} requests", flush=True)
    if args.reuse_output:
        print(
            f"Reused {reuse_records(manifest, args.output, args.reuse_output)} saved responses",
            flush=True,
        )
    if args.prepare_only:
        return
    if args.replay:
        exported = json.loads(args.replay.read_text(encoding="utf-8"))
        if exported["protocol"] != {
            key: value for key, value in manifest.items() if key != "requests"
        }:
            raise ValueError("Exported protocol mismatch")
        expected = {r["request_id"]: r for r in manifest["requests"]}
        records = exported["responses"]
        if len(records) != len(expected) or {r["request_id"] for r in records} != set(
            expected
        ):
            raise ValueError("Exported response set mismatch")
        for record in records:
            validate_record(record, expected[record["request_id"]])
        for record in records:
            save(args.output / "responses" / (record["request_id"] + ".json"), record)
    elif not args.summarize_only:
        asyncio.run(run(manifest, args.output, args.concurrency))
    summarize(manifest, args.output)


if __name__ == "__main__":
    main()
