import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reproduce_leaderboard/methods/015_gpt56_gpt6_comparison.py"
)
spec = importlib.util.spec_from_file_location("gpt_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


@pytest.fixture(scope="module")
def manifest():
    return comparison.build_manifest()


def test_manifest_preserves_dataset_and_comparison_conditions(manifest):
    dataset = comparison.load_default_dataset_for_llm(wordlist_size=100)
    assert manifest["evaluation"]["request_count"] == 600
    assert manifest["evaluation"]["job_count"] == 4
    assert len({r["request_id"] for r in manifest["requests"]}) == 600
    assert manifest["dataset"]["commit"] == comparison.DATASET_COMMIT
    assert manifest["dataset"]["sha256"] == comparison.digest(manifest["queries"])
    for index, query in enumerate(dataset.queries):
        saved = manifest["queries"][index]
        assert saved["candidate_words"] == query.wordlist
        assert saved["positive_words"] == query.positive_words
        request_bodies = [
            manifest["requests"][model_index * 150 + index]["body"]
            for model_index in range(4)
        ]
        assert [body["model"] for body in request_bodies] == list(comparison.MODELS)
        assert all(
            body["messages"] == request_bodies[0]["messages"] for body in request_bodies
        )
        user = request_bodies[0]["messages"][1]["content"]
        assert (
            "\n".join(f"{i}. {word}" for i, word in enumerate(query.wordlist)) in user
        )


def test_request_hashes_schema_and_reservations(manifest):
    total = 0
    for request in manifest["requests"]:
        body = request["body"]
        assert request["body_sha256"] == comparison.digest(body)
        assert (
            request["query_sha256"]
            == manifest["queries"][request["query_index"]]["sha256"]
        )
        assert body["reasoning_effort"] == "medium"
        assert body["max_completion_tokens"] == 24_000
        schema = body["response_format"]["json_schema"]
        assert schema["strict"] is True
        assert schema["schema"]["additionalProperties"] is False
        assert "thoughts" not in schema["schema"]["properties"]
        reservation = request["token_reservation"]
        assert reservation["input"] == len(comparison.canonical_bytes(body)) + 256
        assert reservation["completion"] == 24_000
        assert reservation["total"] == reservation["input"] + 24_000
        total += reservation["total"]
    assert manifest["token_reservation"]["total"] == total
    assert manifest["free_quota_eligibility"] == "unverified"


def test_prompt_matches_existing_baseline(manifest):
    baseline_spec = importlib.util.spec_from_file_location(
        "gpt54_baseline",
        SCRIPT.with_name("010_03_llm_rerank_gpt54_medium_step_by_step.py"),
    )
    baseline = importlib.util.module_from_spec(baseline_spec)
    baseline_spec.loader.exec_module(baseline)
    assert manifest["prompt"]["instructions"] == baseline.PROMPT_INSTRUCTIONS.strip()
    assert (
        manifest["prompt"]["example_suffix"] == baseline.PROMPT_EXAMPLE_SUFFIX.strip()
    )


def test_rejects_unpinned_dataset(monkeypatch):
    monkeypatch.setattr(
        comparison,
        "distribution",
        lambda name: SimpleNamespace(
            read_text=lambda path: '{"vcs_info":{"commit_id":"other"}}'
        ),
    )
    with pytest.raises(ValueError, match="pinned commit"):
        comparison.build_manifest()


def test_rejects_incomplete_dataset(monkeypatch):
    monkeypatch.setattr(
        comparison,
        "load_default_dataset_for_llm",
        lambda **kwargs: SimpleNamespace(queries=[]),
    )
    with pytest.raises(ValueError, match="150 queries"):
        comparison.build_manifest()


def test_cli_writes_explicit_path_and_refuses_overwrite(
    tmp_path, monkeypatch, manifest, capsys
):
    monkeypatch.setattr(comparison, "build_manifest", lambda: manifest)
    output = tmp_path / "prepared.json"
    comparison.main(["--output", str(output)])
    assert json.loads(output.read_text()) == manifest
    assert "150 queries x 4 models = 600 requests" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        comparison.main(["--output", str(output)])
