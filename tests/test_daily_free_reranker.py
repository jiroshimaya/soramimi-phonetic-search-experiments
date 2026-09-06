from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

METHODS = Path(__file__).resolve().parents[1] / "reproduce_leaderboard/methods"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("daily_free", METHODS / "common/daily_free_reranker.py")
NOW = datetime(2026, 9, 5, 12, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def manifest():
    plan = load_module(
        "comparison_plan", METHODS / "015_gpt56_gpt6_comparison.py"
    ).build_manifest()
    plan["requests"].sort(key=lambda request: request["query_index"])
    return plan


@pytest.fixture
def baseline(monkeypatch):
    monkeypatch.setattr(runner, "utc_now", lambda: NOW)
    return {
        "date": "2026-09-05",
        "verified_at": NOW.isoformat(),
        "organization_id": "org-test",
        "project_id": "proj-test",
        "sharing_enabled": True,
        "eligible_models": list(runner.ELIGIBLE),
        "groups": {
            group: {"cap": cap, "used": 0} for group, cap in runner.CAPS.items()
        },
    }


def client(callback):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=callback))
    )


def response(model="gpt-5.6-luna"):
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps({"reranked": list(range(10))})},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    }


def test_reserves_before_sending_and_never_calls_gpt6(manifest, baseline, tmp_path):
    checkpoint = tmp_path / "state.json"
    models = []

    def create(**body):
        saved = json.loads(checkpoint.read_text())
        pending = [
            record
            for record in saved["requests"].values()
            if record["status"] == "pending"
        ]
        assert len(pending) == 1
        assert pending[0]["reservation"] == len(runner.canonical(body)) + 1024 + 24_000
        assert (
            saved["days"][baseline["date"]]["accounted_usage"][pending[0]["group"]]
            >= pending[0]["reservation"]
        )
        assert body["service_tier"] == "default"
        models.append(body["model"])
        return SimpleNamespace(model_dump=lambda **kwargs: response(body["model"]))

    result = runner.run(
        manifest,
        baseline,
        checkpoint,
        tmp_path / "results",
        client(create),
        max_requests=3,
    )
    assert set(models) == set(runner.ELIGIBLE)
    assert result["excluded_models"] == ["gpt-6-astra"]
    state = json.loads(checkpoint.read_text())
    assert state["days"][baseline["date"]]["accounted_usage"] == {
        "premium": 120,
        "small": 240,
    }
    assert all(row["actual_tokens"] == 120 for row in state["requests"].values())
    assert not (tmp_path / "results").exists()
    assert len(list((tmp_path / "state_responses").glob("*.json"))) == 3


def test_waits_without_calls_when_reservations_cannot_fit(manifest, baseline, tmp_path):
    for group in baseline["groups"].values():
        group["used"] = group["cap"] - 24_000
    result = runner.run(
        manifest,
        baseline,
        tmp_path / "state.json",
        tmp_path / "results",
        client(lambda **kwargs: pytest.fail("Quota exceeded")),
    )
    assert result["status"] == "waiting_next_UTC_reset_and_verified_baseline"
    assert result["sent"] == 0


def test_failure_retains_reservation_and_restart_refuses_retry(
    manifest, baseline, tmp_path
):
    checkpoint = tmp_path / "state.json"

    def fail(**kwargs):
        raise TimeoutError("sensitive exception must not be logged")

    with pytest.raises(RuntimeError, match="Manual reconciliation") as error:
        runner.run(manifest, baseline, checkpoint, tmp_path, client(fail))
    assert "sensitive" not in str(error.value)
    state = json.loads(checkpoint.read_text())
    row = next(iter(state["requests"].values()))
    assert row["status"] == "unknown"
    assert (
        state["days"][baseline["date"]]["accounted_usage"][row["group"]]
        == row["reservation"]
    )
    with pytest.raises(ValueError, match="manual reconciliation"):
        runner.run(
            manifest,
            baseline,
            checkpoint,
            tmp_path,
            client(lambda **kwargs: pytest.fail("Retried")),
        )


def test_resume_uses_fresh_org_usage_once_and_skips_completed(
    manifest, baseline, tmp_path
):
    checkpoint = tmp_path / "state.json"
    calls = []

    def create(**kwargs):
        calls.append(kwargs["model"])
        requested_model = kwargs["model"]
        return SimpleNamespace(model_dump=lambda **kwargs: response(requested_model))

    runner.run(manifest, baseline, checkpoint, tmp_path, client(create), max_requests=1)
    with pytest.raises(ValueError, match="rollback"):
        runner.run(
            manifest, baseline, checkpoint, tmp_path, client(create), max_requests=1
        )
    baseline["groups"][runner.ELIGIBLE[calls[0]]]["used"] = 120
    result = runner.run(
        manifest, baseline, checkpoint, tmp_path, client(create), max_requests=1
    )
    assert result["completed"] == 2
    assert calls[0] != calls[1]
    assert (
        sum(
            json.loads(checkpoint.read_text())["days"][baseline["date"]][
                "accounted_usage"
            ].values()
        )
        == 240
    )


@pytest.mark.parametrize("change", ["stale", "day", "eligibility", "cap", "project"])
def test_invalid_baselines_fail_closed(manifest, baseline, tmp_path, change):
    if change == "stale":
        baseline["verified_at"] = (NOW - timedelta(minutes=16)).isoformat()
    elif change == "day":
        baseline["date"] = "2026-09-04"
    elif change == "eligibility":
        baseline["eligible_models"].append("gpt-6-astra")
    elif change == "cap":
        baseline["groups"]["premium"]["cap"] += 1
    else:
        del baseline["project_id"]
    with pytest.raises(ValueError):
        runner.run(
            manifest,
            baseline,
            tmp_path / "state.json",
            tmp_path,
            client(lambda **kwargs: pytest.fail("Sent")),
        )


def test_manifest_change_and_pending_crash_are_not_replayed(
    manifest, baseline, tmp_path
):
    checkpoint = tmp_path / "state.json"

    def crash(**kwargs):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        runner.run(manifest, baseline, checkpoint, tmp_path, client(crash))
    assert (
        next(iter(json.loads(checkpoint.read_text())["requests"].values()))["status"]
        == "pending"
    )
    with pytest.raises(ValueError, match="manual reconciliation"):
        runner.run(manifest, baseline, checkpoint, tmp_path, client(crash))
    altered = deepcopy(manifest)
    altered["prompt"]["variant"] = "different"
    with pytest.raises(ValueError, match="manifest"):
        runner.run(altered, baseline, checkpoint, tmp_path, client(crash))


def test_complete_model_exports_full_macro_recall_only(manifest, baseline, tmp_path):
    state = runner.reconcile_state(manifest, baseline, None)
    model = "gpt-5.6-sol"
    for request in manifest["requests"]:
        if request["body"]["model"] != model:
            continue
        query = manifest["queries"][request["query_index"]]
        ranked = query["positive_words"] + [
            word
            for word in query["candidate_words"]
            if word not in query["positive_words"]
        ]
        state["requests"][request["request_id"]] = {
            "status": "completed",
            "ranked_words": ranked[:10],
            "usage": response()["usage"],
            "actual_tokens": 120,
            "duration_seconds": 1,
        }
    runner.export_complete(manifest, state, tmp_path)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    result = json.loads(files[0].read_text())
    assert len(result["results"]) == 150
    assert result["metrics"]["recall"] == 1.0
    assert result["metrics"]["metadata"]["token_usage"]["total_tokens"] == 18_000


def test_process_lock_rejects_second_runner(tmp_path):
    with runner.exclusive_lock(tmp_path / "state.lock"):
        with pytest.raises(ValueError, match="Another runner"):
            with runner.exclusive_lock(tmp_path / "state.lock"):
                pytest.fail("Concurrent lock acquired")


def test_model_selection_and_returned_model_mismatch(manifest, baseline, tmp_path):
    calls = []

    def create(**body):
        calls.append(body["model"])
        return SimpleNamespace(model_dump=lambda **kwargs: response("gpt-6-astra"))

    with pytest.raises(RuntimeError, match="Manual reconciliation"):
        runner.run(
            manifest,
            baseline,
            tmp_path / "state.json",
            tmp_path,
            client(create),
            max_requests=1,
            models=["gpt-5.6-sol"],
        )
    assert calls == ["gpt-5.6-sol"]
    assert list((tmp_path / "state_responses").glob("*.json"))
    with pytest.raises(ValueError, match="eligible models"):
        runner.run(
            manifest,
            baseline,
            tmp_path / "other.json",
            tmp_path,
            client(create),
            models=["gpt-6-astra"],
        )


def test_stops_before_midnight_without_automatic_reset(
    manifest, baseline, tmp_path, monkeypatch
):
    late = NOW.replace(hour=23, minute=50)
    baseline["verified_at"] = late.isoformat()
    monkeypatch.setattr(runner, "utc_now", lambda: late)
    result = runner.run(
        manifest,
        baseline,
        tmp_path / "state.json",
        tmp_path,
        client(lambda **kwargs: pytest.fail("Sent near reset")),
    )
    assert result["sent"] == 0
    assert result["status"] == "waiting_new_verified_UTC_baseline"


def test_unsettled_reservation_covers_lag_and_can_clear_when_usage_settles(
    manifest, baseline
):
    state = runner.reconcile_state(manifest, baseline, None)
    state["requests"]["completed-small"] = {
        "status": "completed",
        "date": baseline["date"],
        "group": "small",
        "actual_tokens": 120,
    }
    state["days"][baseline["date"]]["accounted_usage"]["small"] = 120
    baseline["groups"]["small"].update(used=60, unsettled_usage_reservation=120)
    runner.validate_baseline(baseline, NOW)
    state = runner.reconcile_state(manifest, baseline, state)
    assert state["days"][baseline["date"]]["accounted_usage"]["small"] == 180
    # The previous conservative bound may decrease as the overlap becomes known.
    baseline["groups"]["small"].update(used=120, unsettled_usage_reservation=0)
    state = runner.reconcile_state(manifest, baseline, state)
    assert state["days"][baseline["date"]]["accounted_usage"]["small"] == 120


def test_observed_usage_regression_fails_even_with_large_reservation(
    manifest, baseline
):
    baseline["groups"]["small"]["used"] = 100
    state = runner.reconcile_state(manifest, baseline, None)
    baseline["groups"]["small"].update(used=99, unsettled_usage_reservation=1000)
    with pytest.raises(ValueError, match="Observed organization usage rollback"):
        runner.reconcile_state(manifest, baseline, state)


@pytest.mark.parametrize("reservation", [-1, True, False, 1.5, "120"])
def test_invalid_unsettled_reservations_fail_closed(baseline, reservation):
    baseline["groups"]["premium"]["unsettled_usage_reservation"] = reservation
    with pytest.raises(ValueError, match="Unexpected daily cap or organization usage"):
        runner.validate_baseline(baseline, NOW)


def test_local_receipts_from_previous_day_do_not_inflate_today(manifest, baseline):
    state = runner.reconcile_state(manifest, baseline, None)
    state["requests"]["yesterday"] = {
        "status": "completed",
        "date": "2026-09-04",
        "group": "premium",
        "actual_tokens": 900_000,
    }
    state = runner.reconcile_state(manifest, baseline, state)
    assert state["days"][baseline["date"]]["accounted_usage"]["premium"] == 0
