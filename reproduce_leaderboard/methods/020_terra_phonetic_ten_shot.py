"""Ten-shot phonetic filler follow-up; saved one-shot controls are rescored offline."""

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random

PATH = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location(
    "filler_transfer_base", PATH.with_name("018_filler_math_transfer.py")
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

MODEL = "gpt-6.1-terra"
CONDITIONS = ["baseline", "counting_100", "dots_100"]
SELECTION_SEED = 20261101
CI_METHOD = "Paired case bootstrap; 20000 draws; seed 20261030; all intervals unadjusted and exploratory"
SOURCE = PATH.parents[1] / "results/019_terra_dot_transfer/phonetic/results.json"


def select_demonstrations(data):
    counts = Counter(case["query"] for case in data["cases"])
    eligible = [
        case
        for case in data["cases"]
        if counts[case["query"]] == 1 and 1 <= len(case["positive_words"]) <= 10
    ]
    return random.Random(SELECTION_SEED).sample(eligible, 9)


def demonstration(case, condition):
    """Show annotated positives, without inventing an unannotated top-ten ranking."""
    positives = set(case["positive_words"])
    indices = [i for i, word in enumerate(case["candidate_words"]) if word in positives]
    if len(indices) != len(positives):
        raise ValueError("Missing or duplicate demonstration positive")
    config = base.get_prompt_config("detailed")
    template = config.user_prompt_template
    filler = base.filler_text(condition)
    if filler:
        template = template.replace("Reranked:", f"Filler: {filler}\nReranked:")
    text = template.format(
        query=case["query"],
        wordlist="\n".join(
            f"{i}. {word}" for i, word in enumerate(case["candidate_words"])
        ),
        topn=len(indices),
    )
    return "Example:\n" + text.strip() + " " + json.dumps({"reranked": indices})


def build_manifest():
    original = base.build_manifest(MODEL, "phonetic", CONDITIONS)
    shots = select_demonstrations(original["data"])
    held_out = {case["query"] for case in shots}
    if "タロウ" in {case["query"] for case in original["data"]["cases"]}:
        raise ValueError("Original synthetic example overlaps evaluation")
    cases = [
        case for case in original["data"]["cases"] if case["query"] not in held_out
    ]
    if len(cases) != 141 or len(held_out) != 9:
        raise ValueError("Expected nine disjoint demonstrations and 141 test queries")
    data = {
        "dataset_commit": original["data"]["dataset_commit"],
        "source_dataset_sha256": original["dataset_sha256"],
        "original_example": base.get_prompt_config("detailed").prompt_example_suffix,
        "additional_demonstrations": shots,
        "cases": cases,
    }
    indices = {case["source_index"] for case in cases}
    requests = []
    for request in original["requests"]:
        if request["source_index"] not in indices:
            continue
        row = deepcopy(request)
        row["body"]["messages"][0]["content"] += "\n\n" + "\n\n".join(
            demonstration(case, row["condition"]) for case in shots
        )
        row["body_sha256"] = base.digest(row["body"])
        requests.append(row)
    random.Random(SELECTION_SEED).shuffle(requests)
    return {
        **original,
        "schema_version": 2,
        "data": data,
        "dataset_sha256": base.digest(data),
        "demonstration_count": 10,
        "selection_seed": SELECTION_SEED,
        "schedule_seed": SELECTION_SEED,
        "primary_comparison": "dots_100 versus baseline (exploratory follow-up)",
        "followup_comparison": "counting_100 versus baseline; 10-shot versus 1-shot",
        "requests": requests,
    }


def one_shot_subset(manifest, output, source=SOURCE):
    exported = json.loads(source.read_text())
    original = base.build_manifest(MODEL, "phonetic", CONDITIONS)
    if exported["protocol"] != {k: v for k, v in original.items() if k != "requests"}:
        raise ValueError("One-shot source protocol mismatch")
    records = {row["request_id"]: row for row in exported["responses"]}
    expected = {row["request_id"] for row in original["requests"]}
    if len(records) != len(exported["responses"]) or set(records) != expected:
        raise ValueError("Incomplete or duplicate one-shot source responses")
    for request in original["requests"]:
        base.validate_record(records[request["request_id"]], request)
    indices = {case["source_index"] for case in manifest["data"]["cases"]}
    requests = [row for row in original["requests"] if row["source_index"] in indices]
    data = {**original["data"], "cases": manifest["data"]["cases"]}
    plan = {
        **original,
        "data": data,
        "dataset_sha256": base.digest(data),
        "demonstration_count": 1,
        "source_export_sha256": base.digest(exported),
        "requests": requests,
    }
    result = base.summarize(
        plan, output, [records[row["request_id"]] for row in requests]
    )
    result["ci_method"] = CI_METHOD
    base.save(output / "summary.json", result)
    return result


def summarize(manifest, output, records=None):
    ten = base.summarize(manifest, output, records)
    ten["ci_method"] = CI_METHOD
    base.save(output / "summary.json", ten)
    one = one_shot_subset(manifest, output / "one_shot_subset")
    if ten["query_order"] != one["query_order"]:
        raise ValueError("One-shot and ten-shot query order differs")
    effects, interactions = {}, {}
    for condition in CONDITIONS:
        old, new = one["cells"][condition]["scores"], ten["cells"][condition]["scores"]
        effects[condition] = base.paired_summary(old, new, False)
        if condition != "baseline":
            old_gain = [
                a - b
                for a, b in zip(old, one["cells"]["baseline"]["scores"], strict=True)
            ]
            new_gain = [
                a - b
                for a, b in zip(new, ten["cells"]["baseline"]["scores"], strict=True)
            ]
            interactions[condition] = base.paired_summary(old_gain, new_gain, False)
    comparison = {
        "n": len(ten["query_order"]),
        "query_order": ten["query_order"],
        "one_shot_source_sha256": json.loads(
            (output / "one_shot_subset/results.json").read_text()
        )["protocol"]["source_export_sha256"],
        "ten_minus_one_shot": effects,
        "change_in_filler_gain": interactions,
        "ci_method": ten["ci_method"],
    }
    base.save(output / "comparison.json", comparison)
    return ten, one, comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--summarize-only", action="store_true")
    mode.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    manifest = build_manifest()
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a new output directory")
    if not path.exists():
        base.save(path, manifest)
    print(f"Prepared {len(manifest['requests'])} new 10-shot requests", flush=True)
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
            asyncio.run(base.execute_requests(manifest, args.output, args.concurrency))
        summarize(manifest, args.output)


if __name__ == "__main__":
    main()
