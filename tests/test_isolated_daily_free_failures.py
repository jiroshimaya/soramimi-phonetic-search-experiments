"""Failure isolation, retry ordering, and accounting across saved attempts."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_daily_free_reranker import client, response, runner
from test_daily_free_reranker import baseline as baseline
from test_daily_free_reranker import manifest as manifest


@pytest.fixture
def pair(manifest, monkeypatch):
    runner.validate_manifest(manifest)
    plan = deepcopy(manifest)
    plan["requests"] = [
        row for row in plan["requests"] if row["body"]["model"] in runner.ELIGIBLE
    ][:2]
    # These scheduler cases intentionally use only two requests from a valid plan.
    monkeypatch.setattr(runner, "validate_manifest", lambda candidate: None)
    return plan


def read_state(tmp_path):
    return json.loads((tmp_path / "state.json").read_text())


def refresh_baseline(baseline, state):
    for group, used in state["days"][baseline["date"]]["accounted_usage"].items():
        baseline["groups"][group]["used"] = used


def execute(plan, baseline, tmp_path, create, **kwargs):
    return runner.run(
        plan,
        baseline,
        tmp_path / "state.json",
        tmp_path / "results",
        client(create),
        **kwargs,
    )


@pytest.mark.parametrize(
    "failure", ["duplicate", "out_of_range", "json", "short", "refused", "truncated"]
)
def test_invalid_ranking_releases_reservation_and_retries_after_peer(
    pair, baseline, tmp_path, failure
):
    first, peer = pair["requests"]
    calls, receipts = [], []

    def create(**body):
        calls.append(body)
        raw = response(body["model"])
        if len(calls) == 1:
            choice = raw["choices"][0]
            if failure == "duplicate":
                choice["message"]["content"] = json.dumps({"reranked": [0] * 10})
            elif failure == "out_of_range":
                choice["message"]["content"] = json.dumps(
                    {"reranked": list(range(91, 101))}
                )
            elif failure == "json":
                choice["message"]["content"] = "{"
            elif failure == "short":
                choice["message"]["content"] = json.dumps({"reranked": [0, 1]})
            elif failure == "refused":
                choice["message"]["refusal"] = "Cannot rank"
            else:
                choice["finish_reason"] = "length"
        if len(calls) == 2:
            saved = read_state(tmp_path)
            failed = saved["requests"][first["request_id"]]
            assert failed["status"] == "output_failed"
            assert failed["actual_tokens"] == 120
            assert failed["usage"] == receipts[0]["usage"]
            pending = sum(
                row["reservation"]
                for row in saved["requests"].values()
                if row["status"] == "pending" and row["group"] == failed["group"]
            )
            assert (
                saved["days"][baseline["date"]]["accounted_usage"][failed["group"]]
                == 120 + pending
            )
        receipts.append(raw)
        return SimpleNamespace(model_dump=lambda **kwargs: raw)

    result = execute(pair, baseline, tmp_path, create)
    assert [body["model"] for body in calls] == [
        first["body"]["model"],
        peer["body"]["model"],
        first["body"]["model"],
    ]
    assert calls[0] == calls[2]  # Retry preserves the exact prompt and parameters.
    assert result["sent"] == 3
    assert result["completed"] == 2
    state = read_state(tmp_path)
    current = state["requests"][first["request_id"]]
    history = state["attempt_history"][first["request_id"]]
    assert current["status"] == "completed"
    assert current["attempt"] == 2
    assert len(history) == 1
    assert history[0]["status"] == "output_failed"
    assert history[0]["attempt"] == 1
    assert Path(history[0]["response_path"]).name == first["request_id"] + ".json"
    assert (
        Path(current["response_path"]).name == first["request_id"] + "__attempt2.json"
    )
    assert json.loads(Path(history[0]["response_path"]).read_text()) == receipts[0]
    assert json.loads(Path(current["response_path"]).read_text()) == receipts[2]
    assert sum(state["days"][baseline["date"]]["accounted_usage"].values()) == 360

    refresh_baseline(baseline, state)
    # A baseline covering only final successful attempts misses the failed receipt.
    baseline["groups"][current["group"]]["used"] -= 120
    with pytest.raises(ValueError, match="rollback"):
        execute(
            pair,
            baseline,
            tmp_path,
            lambda **kwargs: pytest.fail("Usage rollback admitted"),
        )
    refresh_baseline(baseline, state)
    execute(
        pair,
        baseline,
        tmp_path,
        lambda **kwargs: pytest.fail("Completed request retried"),
    )


@pytest.mark.parametrize("failure", ["transport", "model"])
def test_unknown_request_retains_reservation_but_resume_runs_missing_peer(
    pair, baseline, tmp_path, failure
):
    first, peer = pair["requests"]

    def fail(**body):
        if failure == "transport":
            raise TimeoutError("Unconfirmed transport failure")
        raw = response("unexpected-model")
        return SimpleNamespace(model_dump=lambda **kwargs: raw)

    execute(pair, baseline, tmp_path, fail, max_requests=1)
    state = read_state(tmp_path)
    unknown = state["requests"][first["request_id"]]
    assert unknown["status"] == "unknown"
    assert (
        state["days"][baseline["date"]]["accounted_usage"][unknown["group"]]
        == unknown["reservation"]
    )
    # A verified baseline must cover the unresolved reservation too.
    with pytest.raises(ValueError, match="rollback"):
        execute(
            pair,
            baseline,
            tmp_path,
            lambda **kwargs: pytest.fail("Missing reservation admitted"),
        )
    baseline["groups"][unknown["group"]]["unsettled_usage_reservation"] = unknown[
        "reservation"
    ]
    calls = []

    def success(**body):
        calls.append(body)
        return SimpleNamespace(model_dump=lambda **kwargs: response(body["model"]))

    result = execute(pair, baseline, tmp_path, success)
    assert calls == [{**peer["body"], "service_tier": "default"}]
    assert result["completed"] == 1
    assert result["status"] == "needs_review_selected_models"
    saved = read_state(tmp_path)
    assert saved["requests"][first["request_id"]] == unknown
    assert not saved.get("attempt_history", {}).get(first["request_id"])
    assert (
        sum(saved["days"][baseline["date"]]["accounted_usage"].values())
        == unknown["reservation"] + 120
    )
    assert not (tmp_path / "results").exists()


def test_exhausted_output_failure_does_not_export_incomplete_job(
    manifest, baseline, tmp_path
):
    selected = "gpt-5.6-luna"
    requests = [row for row in manifest["requests"] if row["body"]["model"] == selected]
    first = requests[0]
    calls = []

    def create(**body):
        calls.append(body)
        raw = response(body["model"])
        if body["messages"] == first["body"]["messages"]:
            raw["choices"][0]["message"]["content"] = "invalid json"
        return SimpleNamespace(model_dump=lambda **kwargs: raw)

    result = execute(manifest, baseline, tmp_path, create, models=[selected])
    assert len(calls) == 152
    assert calls[:150] == [
        {**row["body"], "service_tier": "default"} for row in requests
    ]
    assert calls[150:] == [calls[0], calls[0]]
    assert result["status"] == "needs_review_selected_models"
    assert result["completed"] == 149
    state = read_state(tmp_path)
    row = state["requests"][first["request_id"]]
    assert row["status"] == "output_failed"
    assert row["attempt"] == 3
    assert [r["attempt"] for r in state["attempt_history"][first["request_id"]]] == [
        1,
        2,
    ]
    assert state["days"][baseline["date"]]["accounted_usage"]["small"] == 152 * 120
    assert not (tmp_path / "results").exists()
    refresh_baseline(baseline, state)
    resumed = execute(
        manifest,
        baseline,
        tmp_path,
        lambda **kwargs: pytest.fail("Exhausted retry sent"),
        models=[selected],
    )
    assert resumed["sent"] == 0
    assert resumed["status"] == "needs_review_selected_models"
    assert not (tmp_path / "results").exists()


def test_retry_reservation_pause_restores_failed_attempt_without_consuming_retry(
    pair, baseline, tmp_path, monkeypatch
):
    from datetime import timedelta
    from test_daily_free_reranker import NOW

    first = pair["requests"][0]

    def fail(**body):
        raw = response(body["model"])
        raw["choices"][0]["message"]["content"] = "{"
        return SimpleNamespace(model_dump=lambda **kwargs: raw)

    execute(pair, baseline, tmp_path, fail, max_requests=1)
    state = read_state(tmp_path)
    # Resume a checkpoint written before explicit attempt numbering existed.
    state["requests"][first["request_id"]].pop("attempt")
    runner.atomic_json(tmp_path / "state.json", state)
    original = deepcopy(state["requests"][first["request_id"]])
    refresh_baseline(baseline, state)
    atomic = runner.atomic_json

    def expire_during_retry_reservation(path, saved):
        atomic(path, saved)
        record = saved.get("requests", {}).get(first["request_id"], {})
        if record.get("status") == "pending" and record.get("attempt") == 2:
            monkeypatch.setattr(runner, "utc_now", lambda: NOW + timedelta(minutes=16))

    monkeypatch.setattr(runner, "atomic_json", expire_during_retry_reservation)
    result = execute(
        pair,
        baseline,
        tmp_path,
        lambda **kwargs: pytest.fail("Expired retry submitted"),
        models=[first["body"]["model"]],
    )
    assert result["sent"] == 0
    assert result["status"] == "waiting_fresh_account_baseline"
    paused = read_state(tmp_path)
    assert paused["requests"][first["request_id"]] == original
    assert not paused.get("attempt_history", {}).get(first["request_id"])
    assert (
        paused["days"][baseline["date"]]["accounted_usage"]
        == state["days"][baseline["date"]]["accounted_usage"]
    )
    assert not list((tmp_path / "state_responses").glob("*__attempt2.json"))

    monkeypatch.setattr(runner, "atomic_json", atomic)
    monkeypatch.setattr(runner, "utc_now", lambda: NOW)
    execute(
        pair,
        baseline,
        tmp_path,
        lambda **body: SimpleNamespace(
            model_dump=lambda **kwargs: response(body["model"])
        ),
        models=[first["body"]["model"]],
    )
    resumed = read_state(tmp_path)
    assert resumed["requests"][first["request_id"]]["attempt"] == 2
    assert resumed["attempt_history"][first["request_id"]] == [original]
    assert sum(resumed["days"][baseline["date"]]["accounted_usage"].values()) == 240
