"""Resume synchronous evaluations against externally verified daily free usage.

The caller must verify enrollment, eligible models and organization-wide usage.
Reservations use UTF-8 bytes plus framing, not a proven provider token bound.
Pending or ambiguous calls require manual reconciliation; they are never retried.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ELIGIBLE = {
    "gpt-5.6-sol": "premium",
    "gpt-5.6-terra": "small",
    "gpt-5.6-luna": "small",
}
CAPS = {"premium": 1_000_000, "small": 10_000_000}
PIN = "6072e13bed37dd2f8eb780e61b6154d21cff2e31"


def utc_now():
    return datetime.now(timezone.utc)


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha256(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def exclusive_lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another runner holds the checkpoint lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_manifest(manifest):
    if manifest["dataset"]["commit"] != PIN or len(manifest["queries"]) != 150:
        raise ValueError("Expected pinned full 150-query manifest")
    if manifest["dataset"]["sha256"] != sha256(manifest["queries"]):
        raise ValueError("Dataset hash mismatch")
    for index, query in enumerate(manifest["queries"]):
        if query["query_index"] != index or len(query["candidate_words"]) != 100:
            raise ValueError("Query index/candidate count mismatch")
        if query["sha256"] != sha256({k: v for k, v in query.items() if k != "sha256"}):
            raise ValueError("Query hash mismatch")
    seen = set()
    counts = {model: 0 for model in ELIGIBLE}
    for request in manifest["requests"]:
        body = request["body"]
        if request["body_sha256"] != sha256(body):
            raise ValueError("Request hash mismatch")
        if body["model"] not in ELIGIBLE:
            continue
        query = manifest["queries"][request["query_index"]]
        identity = (body["model"], request["query_index"])
        if identity in seen or request["query_sha256"] != query["sha256"]:
            raise ValueError("Duplicate or mismatched request")
        seen.add(identity)
        counts[body["model"]] += 1
        if set(body) != {
            "model",
            "messages",
            "reasoning_effort",
            "max_completion_tokens",
            "response_format",
        }:
            raise ValueError("Unexpected request parameters")
        if (
            body["reasoning_effort"] != "medium"
            or body["max_completion_tokens"] != 24_000
        ):
            raise ValueError("Expected medium reasoning and 24000 completion cap")
        schema = body["response_format"].get("json_schema", {})
        if (
            body["response_format"].get("type") != "json_schema"
            or schema.get("strict") is not True
        ):
            raise ValueError("Expected strict structured output")
    if any(count != 150 for count in counts.values()):
        raise ValueError("Expected 150 requests per eligible model")
    ids = [request["request_id"] for request in manifest["requests"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate request identifiers")


def validate_baseline(baseline, now):
    verified = datetime.fromisoformat(baseline["verified_at"].replace("Z", "+00:00"))
    if verified.tzinfo is None or not 0 <= (now - verified).total_seconds() <= 900:
        raise ValueError(
            "Organization usage must be freshly verified within 15 minutes"
        )
    if (
        baseline["date"] != now.date().isoformat()
        or verified.astimezone(timezone.utc).date() != now.date()
    ):
        raise ValueError("A new verified UTC-day organization baseline is required")
    if (
        not baseline.get("organization_id")
        or not baseline.get("project_id")
        or baseline.get("sharing_enabled") is not True
    ):
        raise ValueError("Verified organization and enabled sharing are required")
    if set(baseline["eligible_models"]) != set(ELIGIBLE):
        raise ValueError(
            "Free eligibility must match the three approved GPT-5.6 models"
        )
    if set(baseline["groups"]) != set(CAPS):
        raise ValueError("Both free token groups are required")
    for group, cap in CAPS.items():
        values = baseline["groups"][group]
        if (
            values["cap"] != cap
            or type(values["used"]) is not int
            or not 0 <= values["used"] <= cap
        ):
            raise ValueError("Unexpected daily cap or organization usage")


def reconcile_state(manifest, baseline, state):
    if state is None:
        state = {
            "manifest_sha256": sha256(manifest),
            "organization_id": baseline["organization_id"],
            "project_id": baseline["project_id"],
            "days": {},
            "requests": {},
        }
    if (
        state["manifest_sha256"] != sha256(manifest)
        or state["organization_id"] != baseline["organization_id"]
        or state["project_id"] != baseline["project_id"]
    ):
        raise ValueError("Checkpoint manifest or organization mismatch")
    if any(row["status"] != "completed" for row in state["requests"].values()):
        raise ValueError(
            "Unresolved request requires manual reconciliation; nothing retried"
        )
    if state["days"] and baseline["date"] < max(state["days"]):
        raise ValueError("UTC day rollback refused")
    previous = state["days"].get(baseline["date"])
    used = {group: baseline["groups"][group]["used"] for group in CAPS}
    if previous and any(used[g] < previous["accounted_usage"][g] for g in CAPS):
        raise ValueError(
            "Organization usage rollback refused; refresh usage after accounting settles"
        )
    state["days"][baseline["date"]] = {"baseline": baseline, "accounted_usage": used}
    return state


def parsed_result(raw, query):
    usage = raw["usage"]
    prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
    if any(type(value) is not int or value < 0 for value in (prompt, completion)):
        raise ValueError("Invalid response token usage")
    choice = raw["choices"][0]
    if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
        raise ValueError("Incomplete or refused response")
    indices = json.loads(choice["message"]["content"])["reranked"]
    if (
        len(indices) != 10
        or len(set(indices)) != 10
        or any(type(i) is not int or not 0 <= i < 100 for i in indices)
    ):
        raise ValueError("Expected ten distinct valid candidate indices")
    return [query["candidate_words"][i] for i in indices], prompt + completion


def export_complete(manifest, state, output_dir):
    from soramimi_phonetic_search_dataset.evaluate import calculate_recall

    for model in ELIGIBLE:
        requests = sorted(
            (r for r in manifest["requests"] if r["body"]["model"] == model),
            key=lambda r: r["query_index"],
        )
        records = [state["requests"].get(r["request_id"]) for r in requests]
        if len(records) != 150 or any(
            not row or row["status"] != "completed" for row in records
        ):
            continue
        rows = [
            {
                "query": q["query"],
                "ranked_words": row["ranked_words"],
                "positive_words": q["positive_words"],
            }
            for q, row in zip(manifest["queries"], records, strict=True)
        ]
        result = {
            "parameters": {
                "topn": 10,
                "rank_func": "llm_rerank",
                "execution_timestamp": utc_now().isoformat(),
                "metadata": {
                    "rerank_model_name": model,
                    "rerank_reasoning_effort": "medium",
                    "rerank_prompt_template": "step_by_step",
                    "rerank_input_size": 100,
                    "rerank_backend": "openai_sync",
                    "dataset_commit": PIN,
                    "manifest_sha256": state["manifest_sha256"],
                },
            },
            "metrics": {
                "recall": calculate_recall(
                    [r["ranked_words"] for r in rows],
                    [r["positive_words"] for r in rows],
                    topn=10,
                ),
                "execution_time": sum(r["duration_seconds"] for r in records),
                "metadata": {
                    "token_usage": {
                        "input_tokens": sum(
                            r["usage"]["prompt_tokens"] for r in records
                        ),
                        "completion_tokens": sum(
                            r["usage"]["completion_tokens"] for r in records
                        ),
                        "total_tokens": sum(r["actual_tokens"] for r in records),
                    }
                },
            },
            "results": rows,
        }
        atomic_json(Path(output_dir) / f"015_{model}_medium_step_by_step.json", result)


def run(
    manifest, baseline, checkpoint, output_dir, client, max_requests=None, models=None
):
    checkpoint = Path(checkpoint)
    selected = set(ELIGIBLE if models is None else models)
    if not selected or not selected.issubset(ELIGIBLE):
        raise ValueError("Select only approved eligible models")
    validate_manifest(manifest)
    validate_baseline(baseline, utc_now())
    with exclusive_lock(checkpoint.with_suffix(".lock")):
        state = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
        state = reconcile_state(manifest, baseline, state)
        atomic_json(checkpoint, state)
        sent = 0
        waiting = set()
        status = "complete_eligible_models"
        for request in manifest["requests"]:
            model = request["body"]["model"]
            if model not in selected or request["request_id"] in state["requests"]:
                continue
            if max_requests is not None and sent >= max_requests:
                status = "paused_request_limit"
                break
            now = utc_now()
            reset = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            if (
                now.date().isoformat() != baseline["date"]
                or (reset - now).total_seconds() <= 660
            ):
                status = "waiting_new_verified_UTC_baseline"
                break
            group = ELIGIBLE[model]
            body = {**request["body"], "service_tier": "default"}
            reservation = len(canonical(body)) + 1024 + body["max_completion_tokens"]
            day = state["days"][baseline["date"]]
            if day["accounted_usage"][group] + reservation > CAPS[group]:
                waiting.add(group)
                continue
            record = {
                "status": "pending",
                "date": baseline["date"],
                "model": model,
                "group": group,
                "reservation": reservation,
                "started_at": now.isoformat(),
            }
            state["requests"][request["request_id"]] = record
            day["accounted_usage"][group] += reservation
            atomic_json(checkpoint, state)
            try:
                started = time.monotonic()
                response = client.chat.completions.create(**body)
                duration = time.monotonic() - started
                raw = response.model_dump(mode="json")
                response_path = (
                    checkpoint.parent
                    / (checkpoint.stem + "_responses")
                    / (request["request_id"] + ".json")
                )
                atomic_json(response_path, raw)
                record["response_path"] = str(response_path)
                if raw.get("model") != model and not str(
                    raw.get("model", "")
                ).startswith(model + "-"):
                    raise ValueError("Returned model differs from requested model")
                ranked, actual = parsed_result(
                    raw, manifest["queries"][request["query_index"]]
                )
                if actual > reservation:
                    raise ValueError(
                        "Actual token usage exceeded conservative reservation"
                    )
            except Exception as exc:
                record.update(status="unknown", error_type=type(exc).__name__)
                atomic_json(checkpoint, state)
                raise RuntimeError(
                    "Request unresolved; reserved quota retained. Manual reconciliation required."
                ) from None
            record.update(
                status="completed",
                ranked_words=ranked,
                actual_tokens=actual,
                usage=raw["usage"],
                duration_seconds=duration,
            )
            day["accounted_usage"][group] -= reservation - actual
            atomic_json(checkpoint, state)
            sent += 1
            print(
                json.dumps(
                    {
                        "request_id": request["request_id"],
                        "actual_tokens": actual,
                        "completed": sum(
                            r["status"] == "completed"
                            for r in state["requests"].values()
                        ),
                    }
                ),
                flush=True,
            )
        if waiting and status == "complete_eligible_models":
            status = "waiting_next_UTC_reset_and_verified_baseline"
        elif status == "complete_eligible_models" and selected != set(ELIGIBLE):
            status = "complete_selected_models"
        state["status"] = status
        state["waiting_groups"] = sorted(waiting)
        state["excluded_models"] = sorted(
            {r["body"]["model"] for r in manifest["requests"]} - set(ELIGIBLE)
        )
        atomic_json(checkpoint, state)
        export_complete(manifest, state, output_dir)
        return {
            "status": status,
            "sent": sent,
            "completed": sum(
                r["status"] == "completed" for r in state["requests"].values()
            ),
            "waiting_groups": sorted(waiting),
            "excluded_models": state["excluded_models"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--models", choices=list(ELIGIBLE), nargs="+")
    args = parser.parse_args()
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")
    from openai import OpenAI

    baseline = json.loads(args.baseline.read_text())
    client = OpenAI(
        base_url="https://api.openai.com/v1",
        organization=baseline["organization_id"],
        project=baseline["project_id"],
        max_retries=0,
        timeout=600,
    )
    print(
        json.dumps(
            run(
                json.loads(args.manifest.read_text()),
                baseline,
                args.checkpoint,
                args.output_dir,
                client,
                args.max_requests,
                args.models,
            )
        )
    )


if __name__ == "__main__":
    main()
