from datetime import timedelta
import json
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import pytest

from test_daily_free_reranker import NOW, client, response, runner
from test_daily_free_reranker import baseline as baseline
from test_daily_free_reranker import manifest as manifest


def reservation(request):
    body = {**request["body"], "service_tier": "default"}
    return len(runner.canonical(body)) + 1024 + body["max_completion_tokens"]


def test_workers_overlap_and_every_call_has_a_durable_reservation(
    manifest, baseline, tmp_path
):
    checkpoint = tmp_path / "state.json"
    overlap = Barrier(3)
    observed = []
    guard = Lock()

    def create(**body):
        state = json.loads(checkpoint.read_text())
        row = next(
            row for row in state["requests"].values() if row["model"] == body["model"]
        )
        assert row["status"] == "pending"
        assert (
            state["days"][baseline["date"]]["accounted_usage"][row["group"]]
            >= row["reservation"]
        )
        with guard:
            observed.append(body["model"])
        overlap.wait(timeout=5)
        return SimpleNamespace(model_dump=lambda **kwargs: response(body["model"]))

    result = runner.run(
        manifest,
        baseline,
        checkpoint,
        tmp_path / "results",
        client(create),
        max_requests=3,
        concurrency=3,
    )
    assert set(observed) == set(runner.ELIGIBLE)
    assert result["sent"] == result["completed"] == 3
    state = json.loads(checkpoint.read_text())
    assert state["days"][baseline["date"]]["accounted_usage"] == {
        "premium": 120,
        "small": 240,
    }


def test_shared_pool_waits_then_reuses_released_reservations(
    manifest, baseline, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "state.json"
    small = [
        r
        for r in manifest["requests"]
        if runner.ELIGIBLE.get(r["body"]["model"]) == "small"
    ]
    available = max(reservation(r) for r in small) + 1000
    baseline["groups"]["small"]["used"] = runner.CAPS["small"] - available
    first_can_finish = Event()
    first_started = Event()
    guard = Lock()
    active, peak, calls = 0, 0, 0
    real_wait = runner.wait

    def wait_for_capacity(futures, **kwargs):
        # Four workers are allowed, but only one complete reservation fits.
        assert first_started.wait(timeout=5)
        with guard:
            assert active <= 1
        first_can_finish.set()
        return real_wait(futures, **kwargs)

    monkeypatch.setattr(runner, "wait", wait_for_capacity)

    def create(**body):
        nonlocal active, peak, calls
        with guard:
            active += 1
            peak = max(peak, active)
            calls += 1
        saved = json.loads(checkpoint.read_text())
        assert (
            saved["days"][baseline["date"]]["accounted_usage"]["small"]
            <= runner.CAPS["small"]
        )
        first_started.set()
        assert first_can_finish.wait(timeout=5)
        with guard:
            active -= 1
        return SimpleNamespace(model_dump=lambda **kwargs: response(body["model"]))

    result = runner.run(
        manifest,
        baseline,
        checkpoint,
        tmp_path / "results",
        client(create),
        max_requests=3,
        models=["gpt-5.6-luna", "gpt-5.6-terra"],
        concurrency=4,
    )
    assert peak == 1
    assert calls == result["sent"] == 3
    assert result["waiting_groups"] == []
    saved = json.loads(checkpoint.read_text())
    assert (
        saved["days"][baseline["date"]]["accounted_usage"]["small"]
        == runner.CAPS["small"] - available + 360
    )


def test_premium_capacity_is_independent_of_exhausted_small_pool(
    manifest, baseline, tmp_path
):
    # Small calls cannot fit; two premium calls can overlap independently.
    baseline["groups"]["small"]["used"] = runner.CAPS["small"]
    overlap = Barrier(2)
    calls = []

    def create(**body):
        calls.append(body["model"])
        overlap.wait(timeout=5)
        return SimpleNamespace(model_dump=lambda **kwargs: response(body["model"]))

    result = runner.run(
        manifest,
        baseline,
        tmp_path / "state.json",
        tmp_path / "results",
        client(create),
        max_requests=2,
        concurrency=3,
    )
    assert calls == ["gpt-5.6-sol"] * 2
    assert result["sent"] == 2
    assert result["waiting_groups"] == ["small"]


def test_submission_limit_smaller_than_concurrency_is_exact(
    manifest, baseline, tmp_path
):
    overlap = Barrier(2)
    calls = []

    def create(**body):
        calls.append(body["model"])
        overlap.wait(timeout=5)
        return SimpleNamespace(model_dump=lambda **kwargs: response(body["model"]))

    result = runner.run(
        manifest,
        baseline,
        tmp_path / "state.json",
        tmp_path / "results",
        client(create),
        max_requests=2,
        concurrency=8,
    )
    assert len(calls) == result["sent"] == result["completed"] == 2
    assert result["status"] == "paused_request_limit"


def test_out_of_order_results_stay_aligned_and_resume_skips_both(
    manifest, baseline, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "state.json"
    requests = [
        r for r in manifest["requests"] if r["body"]["model"] in runner.ELIGIBLE
    ]
    first_id, second_id = [r["request_id"] for r in requests[:2]]
    first_model, second_model = [r["body"]["model"] for r in requests[:2]]
    second_persisted = Event()
    real_atomic = runner.atomic_json

    def observe_checkpoint(path, state):
        real_atomic(path, state)
        if path == checkpoint:
            first, second = (
                state["requests"].get(first_id, {}),
                state["requests"].get(second_id, {}),
            )
            if first.get("status") == "pending" and second.get("status") == "completed":
                second_persisted.set()

    monkeypatch.setattr(runner, "atomic_json", observe_checkpoint)

    def create(**body):
        if body["model"] == first_model:
            assert second_persisted.wait(timeout=5)
        raw = response(body["model"])
        offset = 10 if body["model"] == second_model else 0
        raw["choices"][0]["message"]["content"] = json.dumps(
            {"reranked": list(range(offset, offset + 10))}
        )
        raw["usage"]["completion_tokens"] += offset
        return SimpleNamespace(model_dump=lambda **kwargs: raw)

    result = runner.run(
        manifest,
        baseline,
        checkpoint,
        tmp_path / "results",
        client(create),
        max_requests=2,
        concurrency=2,
    )
    assert second_persisted.is_set()
    assert result["completed"] == 2
    state = json.loads(checkpoint.read_text())
    for request, offset in zip(requests[:2], [0, 10], strict=True):
        row = state["requests"][request["request_id"]]
        query = manifest["queries"][request["query_index"]]
        assert row["ranked_words"] == query["candidate_words"][offset : offset + 10]
        assert row["actual_tokens"] == 120 + offset
        assert (
            json.loads(
                (
                    tmp_path / "state_responses" / (request["request_id"] + ".json")
                ).read_text()
            )["model"]
            == request["body"]["model"]
        )
    for group, used in state["days"][baseline["date"]]["accounted_usage"].items():
        baseline["groups"][group]["used"] = used
    called_bodies = []

    def next_create(**body):
        called_bodies.append(body)
        return SimpleNamespace(model_dump=lambda **kwargs: response(body["model"]))

    resumed = runner.run(
        manifest,
        baseline,
        checkpoint,
        tmp_path / "results",
        client(next_create),
        max_requests=2,
        concurrency=2,
    )
    assert resumed["completed"] == 4
    assert {runner.sha256(body) for body in called_bodies} == {
        runner.sha256({**r["body"], "service_tier": "default"}) for r in requests[2:4]
    }


@pytest.mark.parametrize("failure", [TimeoutError, KeyboardInterrupt])
def test_failure_stops_admission_and_drains_successful_peer(
    manifest, baseline, tmp_path, monkeypatch, failure
):
    checkpoint = tmp_path / "state.json"
    selected = [
        r for r in manifest["requests"] if r["body"]["model"] in runner.ELIGIBLE
    ][:2]
    first_model, second_model = [r["body"]["model"] for r in selected]
    peer_started = Event()
    coordinator_waiting = Event()
    failure_observed = Event()
    real_wait = runner.wait
    calls = []

    def observe_completion(futures, **kwargs):
        coordinator_waiting.set()
        result = real_wait(futures, **kwargs)
        if any(f.done() and f.exception() is not None for f in futures):
            failure_observed.set()
        return result

    monkeypatch.setattr(runner, "wait", observe_completion)

    def create(**body):
        calls.append(body["model"])
        if body["model"] == first_model:
            assert peer_started.wait(timeout=5)
            assert coordinator_waiting.wait(timeout=5)
            raise failure("not public")
        assert body["model"] == second_model
        peer_started.set()
        assert failure_observed.wait(timeout=5)
        return SimpleNamespace(model_dump=lambda **kwargs: response(body["model"]))

    expected_error = RuntimeError if failure is TimeoutError else KeyboardInterrupt
    with pytest.raises(expected_error):
        runner.run(
            manifest,
            baseline,
            checkpoint,
            tmp_path / "results",
            client(create),
            max_requests=5,
            concurrency=2,
        )
    assert len(calls) == 2
    state = json.loads(checkpoint.read_text())
    first, second = [state["requests"][r["request_id"]] for r in selected]
    assert first["status"] == ("unknown" if failure is TimeoutError else "pending")
    assert second["status"] == "completed"
    assert (
        state["days"][baseline["date"]]["accounted_usage"][first["group"]]
        >= first["reservation"]
    )
    with pytest.raises(ValueError, match="manual reconciliation"):
        runner.run(
            manifest,
            baseline,
            checkpoint,
            tmp_path / "results",
            client(lambda **kwargs: pytest.fail("Retried unresolved call")),
            concurrency=2,
        )


@pytest.mark.parametrize("boundary", ["midnight", "stale"])
def test_inflight_calls_drain_at_admission_time_boundary(
    manifest, baseline, tmp_path, monkeypatch, boundary
):
    clock = [NOW]
    monkeypatch.setattr(runner, "utc_now", lambda: clock[0])
    overlap = Barrier(2)
    calls = []

    def create(**body):
        calls.append(body["model"])
        overlap.wait(timeout=5)
        clock[0] = (
            NOW.replace(hour=23, minute=50)
            if boundary == "midnight"
            else NOW + timedelta(seconds=901)
        )
        return SimpleNamespace(model_dump=lambda **kwargs: response(body["model"]))

    result = runner.run(
        manifest,
        baseline,
        tmp_path / "state.json",
        tmp_path / "results",
        client(create),
        max_requests=10,
        concurrency=2,
    )
    assert len(calls) == result["sent"] == result["completed"] == 2
    assert result["status"] == (
        "waiting_new_verified_UTC_baseline"
        if boundary == "midnight"
        else "waiting_fresh_account_baseline"
    )


@pytest.mark.parametrize("concurrency", [0, -1, 17, True, 1.5, "2"])
def test_invalid_concurrency_fails_before_sending(
    manifest, baseline, tmp_path, concurrency
):
    with pytest.raises(ValueError, match="Concurrency"):
        runner.run(
            manifest,
            baseline,
            tmp_path / "state.json",
            tmp_path,
            client(lambda **kwargs: pytest.fail("Invalid concurrency sent")),
            concurrency=concurrency,
        )


def test_peer_failure_during_reservation_write_prevents_submission(
    manifest, baseline, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "state.json"
    allow_failure = Event()
    futures = []
    real_executor = runner.ThreadPoolExecutor
    real_atomic = runner.atomic_json
    calls = []

    class ObservedExecutor(real_executor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            futures.append(future)
            return future

    def atomic_with_peer_failure(path, state):
        real_atomic(path, state)
        if path == checkpoint and len(state["requests"]) == 2:
            allow_failure.set()
            with pytest.raises(TimeoutError):
                futures[0].result(timeout=5)

    monkeypatch.setattr(runner, "ThreadPoolExecutor", ObservedExecutor)
    monkeypatch.setattr(runner, "atomic_json", atomic_with_peer_failure)

    def create(**body):
        calls.append(body["model"])
        assert allow_failure.wait(timeout=5)
        raise TimeoutError("provider failure")

    with pytest.raises(RuntimeError, match="Manual reconciliation"):
        runner.run(
            manifest,
            baseline,
            checkpoint,
            tmp_path / "results",
            client(create),
            concurrency=2,
        )
    assert len(calls) == 1
    saved = json.loads(checkpoint.read_text())
    assert len(saved["requests"]) == 1
    row = next(iter(saved["requests"].values()))
    assert row["status"] == "unknown"
    assert (
        sum(saved["days"][baseline["date"]]["accounted_usage"].values())
        == row["reservation"]
    )


def test_reservation_is_released_if_baseline_expires_while_persisting(
    manifest, baseline, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "state.json"
    clock = [NOW]
    real_atomic = runner.atomic_json
    monkeypatch.setattr(runner, "utc_now", lambda: clock[0])

    def expire_during_write(path, state):
        real_atomic(path, state)
        if path == checkpoint and state["requests"]:
            clock[0] = NOW + timedelta(seconds=901)

    monkeypatch.setattr(runner, "atomic_json", expire_during_write)
    result = runner.run(
        manifest,
        baseline,
        checkpoint,
        tmp_path / "results",
        client(lambda **kwargs: pytest.fail("Sent stale request")),
        concurrency=2,
    )
    assert result["status"] == "waiting_fresh_account_baseline"
    assert result["sent"] == 0
    saved = json.loads(checkpoint.read_text())
    assert saved["requests"] == {}
    assert sum(saved["days"][baseline["date"]]["accounted_usage"].values()) == 0


def test_coordinator_interrupt_while_draining_persists_all_worker_results(
    manifest, baseline, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "state.json"
    released = Event()
    real_wait = runner.wait
    interrupted = False

    def interrupt_wait(futures, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            released.set()
            raise KeyboardInterrupt()
        return real_wait(futures, **kwargs)

    monkeypatch.setattr(runner, "wait", interrupt_wait)

    def create(**body):
        assert released.wait(timeout=5)
        return SimpleNamespace(model_dump=lambda **kwargs: response(body["model"]))

    with pytest.raises(KeyboardInterrupt):
        runner.run(
            manifest,
            baseline,
            checkpoint,
            tmp_path / "results",
            client(create),
            max_requests=2,
            concurrency=2,
        )
    rows = json.loads(checkpoint.read_text())["requests"].values()
    assert len(rows) == 2
    assert all(row["status"] == "completed" for row in rows)
