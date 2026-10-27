import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reproduce_leaderboard/methods/017_filler_tokens.py"
)
spec = importlib.util.spec_from_file_location("filler_tokens", SCRIPT)
filler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filler)


@pytest.fixture(scope="module")
def manifest():
    return filler.build_manifest()


def test_paired_requests_only_change_filler(manifest):
    assert len(manifest["requests"]) == 1350
    assert len({r["request_id"] for r in manifest["requests"]}) == 1350
    by_key = {
        (r["condition"], r["repeat"], r["query_index"]): r for r in manifest["requests"]
    }
    for query in manifest["queries"]:
        index = query["query_index"]
        for repeat in range(3):
            base = by_key["baseline", repeat, index]["body"]
            assert base["reasoning_effort"] == "none"
            for condition, text in filler.FILLERS.items():
                request = by_key[condition, repeat, index]
                body = request["body"]
                assert request["body_sha256"] == filler.digest(body)
                assert body["messages"][0] == base["messages"][0]
                restored = body["messages"][1]["content"]
                if text:
                    restored = restored.replace(text + "\nReranked:", "Reranked:")
                assert restored == base["messages"][1]["content"]
                assert "positive" not in restored
                assert (
                    "\n".join(
                        f"{i}. {w}" for i, w in enumerate(query["candidate_words"])
                    )
                    in restored
                )
    assert manifest["fillers"]["dots_300_chars"]["tokens"] == 6
    assert manifest["fillers"]["dots_300_tokens"]["tokens"] == 300


@pytest.mark.parametrize(
    "content",
    [
        "0,1,2,3,4,5,6,7,8,8",
        "0,1,2,3,4,5,6,7,8,100",
        "0,1",
        "Because 0,1,2,3,4,5,6,7,8,9",
        "-1,1,2,3,4,5,6,7,8,9",
    ],
)
def test_invalid_rankings_are_rejected(content):
    with pytest.raises(ValueError):
        filler.parse_indices(content)


def test_score_and_request_validation():
    indices = filler.parse_indices("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")
    assert (
        filler.score(
            indices, {"candidate_words": list(range(20)), "positive_words": [0, 19]}
        )
        == 0.5
    )
    record = {
        "request_id": "a",
        "body_sha256": "x",
        "content": ",".join(map(str, indices)),
        "indices": indices,
        "finish_reason": "stop",
        "valid_ranking": True,
    }
    assert (
        filler.validate_record(record, {"request_id": "a", "body_sha256": "x"})
        == indices
    )
    with pytest.raises(ValueError, match="mismatch"):
        filler.validate_record(record, {"request_id": "a", "body_sha256": "y"})


def test_bootstrap_pairs_by_query_instead_of_treating_repeats_as_independent():
    base = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    treatment = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    result = filler.paired_stats(base, treatment)
    assert result["delta"] == 0
    assert result["improved_queries"] == 1
    assert result["worsened_queries"] == 1
    assert result["unchanged_queries"] == 1
    assert result["ci95"][0] < 0 < result["ci95"][1]
    same = filler.paired_stats(base, base)
    assert same["ci95"] == [0, 0]
    assert filler.paired_stats(np.array(base)[:1], np.array(treatment)[:1]) == result


def test_summarize_requires_all_saved_answers(manifest, tmp_path):
    (tmp_path / "responses").mkdir()
    with pytest.raises(FileNotFoundError):
        filler.summarize(manifest, tmp_path)


def test_replay_recomputes_complete_results_without_api(
    manifest, tmp_path, monkeypatch
):
    import json

    content = ", ".join(map(str, range(10)))
    exported = {
        "protocol": {k: v for k, v in manifest.items() if k != "requests"},
        "responses": [
            {
                "request_id": r["request_id"],
                "body_sha256": r["body_sha256"],
                "model": "test-snapshot",
                "content": content,
                "indices": list(range(10)),
                "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                "finish_reason": "stop",
                "valid_ranking": True,
            }
            for r in manifest["requests"]
        ],
    }
    source = tmp_path / "export.json"
    source.write_text(json.dumps(exported))
    output = tmp_path / "recomputed"
    monkeypatch.setattr(
        filler,
        "build_manifest",
        lambda repeats, structured=False, counting=False: manifest,
    )
    monkeypatch.setattr(
        filler.asyncio, "run", lambda coro: pytest.fail("Replay must not call API")
    )
    filler.main(["--output", str(output), "--replay", str(source)])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["request_count"] == 1350
    assert summary["cells"]["baseline"]["input_tokens"] == 45000
    for condition in ("dots_300_chars", "dots_300_tokens"):
        assert summary["cells"][condition]["versus_baseline"]["ci95"] == [0, 0]
    exported["responses"][0]["body_sha256"] = "tampered"
    source.write_text(json.dumps(exported))
    with pytest.raises(ValueError, match="mismatch"):
        filler.main(["--output", str(tmp_path / "invalid"), "--replay", str(source)])


def test_invalid_first_answer_is_retained_and_scored_zero():
    record = {
        "request_id": "a",
        "body_sha256": "x",
        "content": "...0,1,2,3,4,5,6,7,8,9",
        "finish_reason": "stop",
        "valid_ranking": False,
        "indices": [],
        "parse_error": "Response must contain only comma-separated indices",
    }
    indices = filler.validate_record(record, {"request_id": "a", "body_sha256": "x"})
    assert indices == []
    assert (
        filler.score(
            indices, {"candidate_words": list(range(20)), "positive_words": [0, 19]}
        )
        == 0
    )
    record["content"] = "0,1,2,3,4,5,6,7,8,9"
    with pytest.raises(ValueError, match="mismatch"):
        filler.validate_record(record, {"request_id": "a", "body_sha256": "x"})


def test_structured_mode_preserves_requests_except_response_schema(manifest):
    structured = filler.build_manifest(structured=True)
    assert structured["queries"] == manifest["queries"]
    assert structured["output_format"] == "json_schema"
    for plain, fixed in zip(manifest["requests"], structured["requests"], strict=True):
        assert plain["request_id"] == fixed["request_id"]
        body = dict(fixed["body"])
        schema = body.pop("response_format")["json_schema"]
        assert schema["strict"] is True
        assert set(schema["schema"]["properties"]) == {"reranked"}
        assert body == plain["body"]
        assert fixed["body_sha256"] == filler.digest(fixed["body"])
    content = '{"reranked": [0,1,2,3,4,5,6,7,8,9]}'
    assert filler.parse_answer(content, "stop", structured=True) == list(range(10))
    with pytest.raises(ValueError):
        filler.parse_answer(
            '{"reranked": [true,1,2,3,4,5,6,7,8,9]}', "stop", structured=True
        )


def test_counting_condition_reuses_exact_baseline_protocol(manifest):
    counting = filler.build_manifest(structured=True, counting=True)
    assert len(counting["requests"]) == 900
    assert set(counting["fillers"]) == {"baseline", "counting_300"}
    assert counting["fillers"]["counting_300"]["text"].split() == list(
        map(str, range(1, 301))
    )
    assert counting["fillers"]["counting_300"]["tokens"] > 300
    structured = filler.build_manifest(structured=True)
    baseline = {
        r["request_id"]: r["body_sha256"]
        for r in structured["requests"]
        if r["condition"] == "baseline"
    }
    assert {
        r["request_id"]: r["body_sha256"]
        for r in counting["requests"]
        if r["condition"] == "baseline"
    } == baseline


def test_cache_reuse_preserves_replicates_and_rejects_changed_prompts(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    for directory in (source, target):
        (directory / "responses").mkdir(parents=True)
    record = {
        "request_id": "q__r0",
        "body_sha256": "same",
        "content": "0,1,2,3,4,5,6,7,8,9",
        "indices": list(range(10)),
        "valid_ranking": True,
        "finish_reason": "stop",
    }
    filler.save(source / "responses/q__r0.json", record)
    filler.save(
        source / "responses/q__r1.json",
        {**record, "request_id": "q__r1", "body_sha256": "changed"},
    )
    requests = [{"request_id": f"q__r{i}", "body_sha256": "same"} for i in range(3)]
    assert filler.reuse_records({"requests": requests}, target, source) == 1
    assert (target / "responses/q__r0.json").exists()
    assert not (target / "responses/q__r1.json").exists()
    assert not (target / "responses/q__r2.json").exists()
