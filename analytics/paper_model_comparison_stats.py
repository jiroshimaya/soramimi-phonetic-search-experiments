"""Export complete model comparisons as JSON, Markdown, and LaTeX tables."""

import argparse
from collections import Counter
from importlib.metadata import distribution
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SUBSETS_PATH = ROOT / "reproduce_leaderboard/data/paper_subsets.json"
MODELS = ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra")
EFFORTS = ("none", "medium")
PROMPTS = ("simple", "detailed", "step_by_step")
VARIANTS = ("v1", "v2", "v3", "v4", "v5")
SUBSET_KEYS = ("easy", "medium", "hard", "overall")
PIN = "6072e13bed37dd2f8eb780e61b6154d21cff2e31"


def load_evaluation_data():
    package = distribution("soramimi-phonetic-search-dataset")
    source = json.loads(package.read_text("direct_url.json") or "{}")
    if source.get("vcs_info", {}).get("commit_id") != PIN:
        raise ValueError("The pinned dataset installation is required")
    data = json.loads(
        Path(
            package.locate_file("soramimi_phonetic_search_dataset/data/baseball.json")
        ).read_text(encoding="utf-8")
    )
    subsets = json.loads(SUBSETS_PATH.read_text(encoding="utf-8"))
    queries = data["queries"]
    if len(queries) != 150 or set(subsets["query_to_subset"]) != {
        q["query"] for q in queries
    }:
        raise ValueError("Frozen difficulty mapping does not match the 150 queries")
    if Counter(subsets["query_to_subset"][query["query"]] for query in queries) != {
        "easy": 65,
        "medium": 47,
        "hard": 38,
    }:
        raise ValueError("Unexpected difficulty counts")
    return queries, subsets


def score_trial(result, queries, subsets):
    rows = result["results"]
    if len(rows) != 150 or len(queries) != 150:
        raise ValueError("A trial requires all 150 queries")
    if result["parameters"]["topn"] != 10:
        raise ValueError("Expected Recall@10 results")
    values = {key: [] for key in SUBSET_KEYS}
    for row, query in zip(rows, queries, strict=True):
        if row["query"] != query["query"] or row["positive_words"] != query["positive"]:
            raise ValueError("Trial query order or positive words mismatch")
        ranked = row["ranked_words"]
        candidates = set(
            query["positive"] + query["hard_negatives"][: 100 - len(query["positive"])]
        )
        if (
            len(ranked) != 10
            or len(set(ranked)) != 10
            or not set(ranked).issubset(candidates)
        ):
            raise ValueError("Trial ranking must contain ten distinct candidate words")
        recall = len(set(ranked) & set(query["positive"])) / len(query["positive"])
        values["overall"].append(recall)
        values[subsets["query_to_subset"][query["query"]]].append(recall)
    scores = {key: statistics.mean(items) for key, items in values.items()}
    if abs(result["metrics"]["recall"] - scores["overall"]) > 1e-12:
        raise ValueError("Stored recall differs from recomputed recall")
    return scores


def validate_job_metadata(result, model, effort, prompt, variant):
    metadata = result["parameters"]["metadata"]
    expected = {
        "job_id": f"{model}__{effort}__{prompt}__{variant}",
        "model": model,
        "reasoning_effort": effort,
        "prompt_template": prompt,
        "prompt_variant": variant,
    }
    aliases = {
        "model": "rerank_model_name",
        "reasoning_effort": "rerank_reasoning_effort",
        "prompt_template": "rerank_prompt_template",
        "prompt_variant": "rerank_prompt_variant",
    }
    for key, value in expected.items():
        fields = [key, aliases[key]] if key in aliases else [key]
        found = [metadata[field] for field in fields if field in metadata]
        if not found or any(actual != value for actual in found):
            raise ValueError(f"Trial metadata mismatch: {key}")


def summarize(results_dir, models=MODELS):
    queries, subsets = load_evaluation_data()
    cells = []
    for model in models:
        for effort in EFFORTS:
            for prompt in PROMPTS:
                variants = {}
                missing = []
                for variant in VARIANTS:
                    identifier = f"{model}__{effort}__{prompt}__{variant}"
                    path = Path(results_dir) / f"016_{identifier}.json"
                    if not path.exists():
                        missing.append(variant)
                        continue
                    result = json.loads(path.read_text(encoding="utf-8"))
                    validate_job_metadata(result, model, effort, prompt, variant)
                    variants[variant] = score_trial(result, queries, subsets)
                cell = {
                    "model": model,
                    "reasoning_effort": effort,
                    "prompt_template": prompt,
                    "status": "complete" if not missing else "pending",
                    "completed_variants": len(variants),
                    "missing_variants": missing,
                    "variants": variants,
                }
                if not missing:
                    cell["stats"] = {
                        key: {
                            "mean": statistics.mean(v[key] for v in variants.values()),
                            "stdev": statistics.stdev(
                                v[key] for v in variants.values()
                            ),
                            "n": 5,
                        }
                        for key in SUBSET_KEYS
                    }
                cells.append(cell)
    return {
        "metric": "macro Recall@10",
        "query_count": 150,
        "dataset_commit": PIN,
        "subsets": subsets,
        "standard_deviation_ddof": 1,
        "status": "complete"
        if all(c["status"] == "complete" for c in cells)
        else "pending",
        "completed_cells": sum(c["status"] == "complete" for c in cells),
        "expected_cells": len(cells),
        "cells": cells,
    }


def markdown_table(summary):
    lines = [
        "Recall@10; mean ± sample standard deviation over five wording variants (n = 5).",
        "",
        "| Model | Reasoning | Prompt | Easy (65) | Medium (47) | Hard (38) | Overall | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for cell in summary["cells"]:
        scores = (
            [
                f"{cell['stats'][key]['mean']:.3f} ± {cell['stats'][key]['stdev']:.3f}"
                for key in SUBSET_KEYS
            ]
            if cell["status"] == "complete"
            else ["—"] * 4
        )
        status = (
            "complete"
            if cell["status"] == "complete"
            else f"pending ({cell['completed_variants']}/5 variants)"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    cell["model"],
                    cell["reasoning_effort"],
                    cell["prompt_template"],
                    *scores,
                    status,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def latex_table(summary):
    lines = [
        r"\begin{tabular}{lllcccc l}",
        r"\hline",
        r"Model & Reasoning & Prompt & Easy (65) & Medium (47) & Hard (38) & Overall & Status \\",
        r"\hline",
    ]
    for cell in summary["cells"]:
        scores = (
            [
                f"${cell['stats'][key]['mean']:.3f}\\pm{cell['stats'][key]['stdev']:.3f}$"
                for key in SUBSET_KEYS
            ]
            if cell["status"] == "complete"
            else ["--"] * 4
        )
        status = (
            "complete"
            if cell["status"] == "complete"
            else f"pending ({cell['completed_variants']}/5)"
        )
        labels = [cell["model"], cell["reasoning_effort"], cell["prompt_template"]]
        lines.append(
            " & ".join(
                [*(label.replace("_", r"\_") for label in labels), *scores, status]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            "% Mean and sample standard deviation over five wording variants (ddof=1).",
        ]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "paper_model_comparison.json": json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n",
        "paper_model_comparison.md": markdown_table(summary),
        "paper_model_comparison.tex": latex_table(summary),
    }
    for name, content in artifacts.items():
        (output_dir / name).write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarize(args.results_dir)
    write_artifacts(summary, args.output_dir)
    print(
        f"{summary['completed_cells']}/{summary['expected_cells']} complete cells; tables saved in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
