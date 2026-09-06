from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import statistics

import pytest

ROOT = Path(__file__).resolve().parents[1]
METHODS = ROOT / "reproduce_leaderboard/methods"
RESULTS = ROOT / "reproduce_leaderboard/results"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


plan = module("paper_plan", METHODS / "016_paper_model_comparison.py")
stats = module("paper_stats", ROOT / "analytics/paper_model_comparison_stats.py")


@pytest.fixture(scope="module")
def manifest():
    return plan.build_manifest()


def historical_path(effort, prompt, variant):
    if variant == "v1":
        index = plan.PROMPTS.index(prompt) + 1
        prefix = "008" if effort == "none" else "010"
        suffix = "" if effort == "none" else "medium_"
        return RESULTS / f"{prefix}_0{index}_llm_rerank_gpt54_{suffix}{prompt}.json"
    prefix = "012" if effort == "none" else "013"
    directory = (
        "012_prompt_variations"
        if effort == "none"
        else "013_prompt_variations_reasoning"
    )
    return RESULTS / directory / f"{prefix}_{prompt}_{variant}.json"


def test_full_matrix_preserves_original_dataset_and_reusable_requests(manifest):
    original = plan.base_protocol().build_manifest()
    assert manifest["schema_version"] == 2
    assert manifest["queries"] == original["queries"]
    assert manifest["dataset"] == original["dataset"]
    assert len(manifest["jobs"]) == 90
    assert len(manifest["requests"]) == 13_500
    assert len({r["request_id"] for r in manifest["requests"]}) == 13_500
    assert set(r["body"]["model"] for r in manifest["requests"]) == set(plan.MODELS)
    assert manifest["unavailable_models"][0]["model"] == "gpt-6-astra"
    originals = {
        (r["body"]["model"], r["query_index"]): r for r in original["requests"]
    }
    for request in manifest["requests"][:450]:
        assert "__medium__step_by_step__v1__" in request["request_id"]
        previous = originals[(request["body"]["model"], request["query_index"])]
        assert request["body"] == previous["body"]
        assert request["body_sha256"] == previous["body_sha256"]
        assert request["query_sha256"] == previous["query_sha256"]
    assert [r["body"]["model"] for r in manifest["requests"][:3]] == list(plan.MODELS)
    assert all(r["query_index"] == 0 for r in manifest["requests"][:3])
    assert all(
        job["reasoning_effort"] == "medium" and job["prompt_template"] == "step_by_step"
        for job in manifest["jobs"][:15]
    )
    assert [
        manifest["jobs"][index * 3]["prompt_variant"] for index in range(5)
    ] == list(plan.VARIANTS)


def test_jobs_and_requests_encode_every_condition_explicitly(manifest):
    digest = plan.base_protocol().digest
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    seen = {identifier: [] for identifier in jobs}
    for job in jobs.values():
        fields = {
            key: job[key]
            for key in (
                "prompt_instructions",
                "prompt_example_suffix",
                "user_prompt_template",
            )
        }
        assert job["prompt_sha256"] == digest(fields)
    for request in manifest["requests"]:
        job = jobs[request["job_id"]]
        body = request["body"]
        seen[job["job_id"]].append(request["query_index"])
        assert body["model"] == job["model"]
        assert body["reasoning_effort"] == job["reasoning_effort"]
        assert body["max_completion_tokens"] == (
            1000 if job["reasoning_effort"] == "none" else 24_000
        )
        assert body["response_format"]["json_schema"]["strict"] is True
        assert request["body_sha256"] == digest(body)
        assert (
            request["query_sha256"]
            == manifest["queries"][request["query_index"]]["sha256"]
        )
        assert (
            request["request_id"] == f"{job['job_id']}__q{request['query_index']:04d}"
        )
    assert all(indices == list(range(150)) for indices in seen.values())
    assert len(seen) == 3 * 2 * 3 * 5


def test_variants_reuse_existing_experiment_messages(manifest):
    nonreasoning = module(
        "original_012", METHODS / "012_prompt_variations_nonreasoning_batch.py"
    )
    reasoning = module(
        "original_013", METHODS / "013_prompt_variations_reasoning_batch.py"
    )
    for effort, original in (("none", nonreasoning), ("medium", reasoning)):
        for prompt in plan.PROMPTS:
            for variant in plan.VARIANTS[1:]:
                identifier = plan.job_id(plan.MODELS[0], effort, prompt, variant)
                requests = [
                    r for r in manifest["requests"] if r["job_id"] == identifier
                ]
                expected = original.build_rerank_messages(
                    [q["query"] for q in manifest["queries"]],
                    [q["candidate_words"] for q in manifest["queries"]],
                    topn=10,
                    prompt_template=prompt,
                    prompt_instructions=original.PROMPT_VARIATIONS[prompt][variant],
                    prompt_example_suffix=original.build_example_suffix("none"),
                    input_transform="none",
                )
                assert [r["body"]["messages"] for r in requests] == expected


def test_frozen_subsets_include_duplicate_query_occurrences(manifest):
    mapping = manifest["subsets"]["query_to_subset"]
    assert len(mapping) == 148
    assert Counter(mapping[q["query"]] for q in manifest["queries"]) == {
        "easy": 65,
        "medium": 47,
        "hard": 38,
    }
    assert manifest["queries"][7]["query"] == manifest["queries"][8]["query"] == "アレ"
    assert manifest["queries"][7]["sha256"] != manifest["queries"][8]["sha256"]
    assert (
        manifest["subsets"]["source"]["commit"]
        == "35bf9075f885b74df55ad7a2415861939ccdcf7c"
    )


def test_existing_thirty_trials_reproduce_comparison_numbers():
    queries, subsets = stats.load_evaluation_data()
    expected = {
        ("none", "simple"): (
            "0.888±0.034",
            "0.280±0.037",
            "0.287±0.039",
            "0.545±0.017",
        ),
        ("none", "detailed"): (
            "0.928±0.012",
            "0.284±0.024",
            "0.246±0.028",
            "0.553±0.017",
        ),
        ("none", "step_by_step"): (
            "0.851±0.016",
            "0.246±0.039",
            "0.282±0.029",
            "0.517±0.017",
        ),
        ("medium", "simple"): (
            "0.948±0.018",
            "0.440±0.053",
            "0.347±0.028",
            "0.636±0.023",
        ),
        ("medium", "detailed"): (
            "1.000±0.000",
            "0.827±0.046",
            "0.557±0.077",
            "0.834±0.017",
        ),
        ("medium", "step_by_step"): (
            "0.978±0.014",
            "0.911±0.020",
            "0.899±0.021",
            "0.937±0.004",
        ),
    }
    for (effort, prompt), expected_scores in expected.items():
        trials = [
            stats.score_trial(
                json.loads(historical_path(effort, prompt, variant).read_text()),
                queries,
                subsets,
            )
            for variant in plan.VARIANTS
        ]
        actual = tuple(
            f"{statistics.mean(trial[key] for trial in trials):.3f}±{statistics.stdev(trial[key] for trial in trials):.3f}"
            for key in stats.SUBSET_KEYS
        )
        assert actual == expected_scores


def write_trial(results_dir, variant):
    model, effort, prompt = plan.MODELS[0], "medium", "step_by_step"
    trial = json.loads(historical_path(effort, prompt, variant).read_text())
    identifier = plan.job_id(model, effort, prompt, variant)
    trial["parameters"]["metadata"] = {
        "job_id": identifier,
        "rerank_model_name": model,
        "rerank_reasoning_effort": effort,
        "rerank_prompt_template": prompt,
        "rerank_prompt_variant": variant,
    }
    path = results_dir / f"016_{identifier}.json"
    path.write_text(json.dumps(trial, ensure_ascii=False))
    return path


def test_partial_cells_remain_pending_and_complete_cells_have_sample_sd(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    write_trial(results_dir, "v1")
    pending = stats.summarize(results_dir)
    assert pending["expected_cells"] == 18
    assert pending["completed_cells"] == 0
    assert all("stats" not in cell for cell in pending["cells"])
    assert "pending (1/5 variants)" in stats.markdown_table(pending)
    for variant in plan.VARIANTS[1:]:
        write_trial(results_dir, variant)
    summary = stats.summarize(results_dir)
    assert summary["completed_cells"] == 1
    cell = next(c for c in summary["cells"] if c["status"] == "complete")
    values = [trial["overall"] for trial in cell["variants"].values()]
    assert cell["stats"]["overall"]["stdev"] == statistics.stdev(values)
    assert cell["stats"]["overall"]["stdev"] != statistics.pstdev(values)
    output = tmp_path / "artifacts"
    stats.write_artifacts(summary, output)
    assert len(list(output.iterdir())) == 3
    assert "0.937 ± 0.004" in (output / "paper_model_comparison.md").read_text()
    latex = (output / "paper_model_comparison.tex").read_text()
    assert r"0.937\pm0.004" in latex and r"step\_by\_step" in latex
    assert (
        json.loads((output / "paper_model_comparison.json").read_text())["status"]
        == "pending"
    )


def test_incomplete_trial_and_mismatched_metadata_are_rejected(tmp_path):
    path = write_trial(tmp_path, "v1")
    trial = json.loads(path.read_text())
    broken = deepcopy(trial)
    broken["results"].pop()
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="all 150"):
        stats.summarize(tmp_path)
    trial["parameters"]["metadata"]["rerank_model_name"] = "gpt-6-astra"
    path.write_text(json.dumps(trial))
    with pytest.raises(ValueError, match="metadata mismatch"):
        stats.summarize(tmp_path)


def test_prepare_cli_requires_explicit_output_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(plan, "build_manifest", lambda: {"jobs": [], "requests": []})
    path = tmp_path / "plan.json"
    plan.main(["--output", str(path)])
    assert json.loads(path.read_text()) == {"jobs": [], "requests": []}
    with pytest.raises(FileExistsError):
        plan.main(["--output", str(path)])
