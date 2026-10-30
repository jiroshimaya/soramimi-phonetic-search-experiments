import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reproduce_leaderboard/methods/018_filler_math_transfer.py"
)
spec = importlib.util.spec_from_file_location("math_transfer", SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.fixture(scope="module")
def manifest():
    return study.build_manifest("test-model", "math")


def test_frozen_truth_and_demonstration_exclusion(manifest):
    data = manifest["data"]
    assert len(data["cases"]) == 600
    assert len(data["few_shot"]) == 10
    assert {q["source_index"] for q in data["few_shot"]} == {
        0,
        333,
        667,
        1000,
        1333,
        1667,
        2000,
        2333,
        2667,
        2999,
    }
    assert not {q["source_index"] for q in data["few_shot"]} & {
        q["source_index"] for q in data["cases"]
    }
    assert len({q["source_index"] for q in data["cases"]}) == 600
    for q in data["cases"] + data["few_shot"]:
        assert study.arithmetic_answer(q["problem"]) == q["answer"]
    assert study.arithmetic_answer("Evaluate this Python expression. -7 // 3") == -3
    assert study.arithmetic_answer("Evaluate this Python expression. -7 % 3") == 2
    with pytest.raises(ValueError):
        study.arithmetic_answer("__import__('os').system('false')")


def test_source_prompt_format_and_all_demonstrations_receive_filler(manifest):
    assert len(manifest["requests"]) == 2400
    assert manifest["primary_comparison"] == "counting_100 versus baseline"
    by_key = {(r["condition"], r["source_index"]): r for r in manifest["requests"]}
    case = manifest["data"]["cases"][0]
    for condition in study.CONDITIONS:
        request = by_key[condition, case["source_index"]]
        assert request["body_sha256"] == study.digest(request["body"])
        body = request["body"]
        assert body["reasoning_effort"] == "none"
        assert body["temperature"] == 0.0
        assert body["max_completion_tokens"] == 100
        assert "response_format" not in body
        messages = body["messages"]
        assert len(messages) == 21
        assert [m["role"] for m in messages] == ["user", "assistant"] * 10 + ["user"]
        assert messages[-1]["content"].endswith("\n\nAnswer:")
        assert case["problem"] in messages[-1]["content"]
        for user in messages[::2]:
            if condition == "baseline":
                assert "Filler:" not in user["content"]
            else:
                assert "Filler: " + study.filler_text(condition) in user["content"]
    other = study.build_manifest("other-model", "math")
    assert all(
        a["body_sha256"] != b["body_sha256"]
        for a, b in zip(manifest["requests"], other["requests"], strict=True)
    )


@pytest.mark.parametrize(
    "content,prediction",
    [("-1", -1), ("Answer: 1,234", 1234), ("[Answer: 20]", 20), ("I think -1", None)],
)
def test_math_scoring_preserves_invalid_minus_one_distinction(content, prediction):
    result = study.score(
        {"content": content, "finish_reason": "stop"}, {"answer": -1}, "math"
    )
    assert result["prediction"] == prediction
    assert result["valid"] == (prediction is not None)
    assert result["score"] == float(prediction == -1)


def test_first_bad_answer_is_saved_and_never_retried(tmp_path):
    calls = []
    request = {
        "request_id": "baseline__q1",
        "body_sha256": "x",
        "body": {"model": "test"},
    }
    payload = {
        "model": "test",
        "choices": [
            {"message": {"content": "I cannot answer"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json=payload)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await study.execute_requests(
                {"requests": [request]}, tmp_path, client=client
            )
            await study.execute_requests(
                {"requests": [request]}, tmp_path, client=client
            )

    asyncio.run(exercise())
    assert len(calls) == 1
    assert (
        json.loads((tmp_path / "provider_responses/baseline__q1.json").read_text())
        == payload
    )
    record = json.loads((tmp_path / "responses/baseline__q1.json").read_text())
    assert study.score(record, {"answer": -1}, "math")["score"] == 0
    with pytest.raises(ValueError, match="mismatch"):
        study.validate_record(record, {**request, "body_sha256": "different"})


def test_paired_binary_comparison():
    result = study.paired_summary([0] * 10, [1] * 10, True)
    assert result["delta"] == 1
    assert result["ci95"] == [1, 1]
    assert result["mcnemar_exact_two_sided_p"] == 2 / 1024
    assert study.paired_summary([1, 0], [1, 0], True)["mcnemar_exact_two_sided_p"] == 1


def test_phonetic_target_and_example_both_receive_filler():
    case = {"query": "アオ", "candidate_words": [str(i) for i in range(100)]}
    messages = study.phonetic_messages(case, "counting_100")
    for message in messages:
        assert "Filler: " + study.filler_text("counting_100") in message["content"]
    assert "Filler:" not in study.phonetic_messages(case, "baseline")[0]["content"]
