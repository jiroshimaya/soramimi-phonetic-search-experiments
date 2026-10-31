from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reproduce_leaderboard/methods/020_terra_phonetic_ten_shot.py"
)
spec = importlib.util.spec_from_file_location("phonetic_ten_shot", SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.fixture(scope="module")
def manifest():
    return study.build_manifest()


def test_examples_are_fixed_and_excluded_by_query_and_index(manifest):
    shots = manifest["data"]["additional_demonstrations"]
    cases = manifest["data"]["cases"]
    assert [row["source_index"] for row in shots] == [
        25,
        31,
        102,
        17,
        13,
        14,
        58,
        72,
        86,
    ]
    assert len(cases) == 141
    assert len(manifest["requests"]) == 423
    assert not {row["query"] for row in shots} & {row["query"] for row in cases}
    assert not {row["source_index"] for row in shots} & {
        row["source_index"] for row in cases
    }
    assert "タロウ" not in {row["query"] for row in cases}
    source = study.base.phonetic_data()
    assert sorted(shots + cases, key=lambda row: row["source_index"]) == source["cases"]


def test_only_extra_examples_change_requests_and_filler_copies_match_arithmetic(
    manifest,
):
    original = study.base.build_manifest(study.MODEL, "phonetic", study.CONDITIONS)
    by_id = {row["request_id"]: row for row in original["requests"]}
    for request in manifest["requests"]:
        body = deepcopy(request["body"])
        before = by_id[request["request_id"]]["body"]
        assert body["messages"][0]["content"].startswith(
            before["messages"][0]["content"] + "\n\n"
        )
        assert body["messages"][0]["content"].count("Example:") == 10
        filler = study.base.filler_text(request["condition"])
        copies = sum(
            message["content"].count("Filler: ") for message in body["messages"]
        )
        assert copies == (11 if filler else 0)
        if filler:
            assert (
                sum(
                    message["content"].count("Filler: " + filler)
                    for message in body["messages"]
                )
                == 11
            )
        body["messages"][0]["content"] = before["messages"][0]["content"]
        assert body == before
        assert request["body_sha256"] != by_id[request["request_id"]]["body_sha256"]


def test_demo_answers_use_only_annotated_positives(manifest):
    for case in manifest["data"]["additional_demonstrations"]:
        for condition in study.CONDITIONS:
            text = study.demonstration(case, condition)
            answer = json.loads(text.split("Reranked:", 1)[1])
            indices = answer["reranked"]
            assert {case["candidate_words"][i] for i in indices} == set(
                case["positive_words"]
            )
            assert f"Top N: {len(indices)}" in text


def test_saved_one_shot_subset_is_reused_and_hash_mismatch_rejected(manifest, tmp_path):
    result = study.one_shot_subset(manifest, tmp_path / "valid")
    assert len(result["query_order"]) == 141
    original = json.loads(study.SOURCE.read_text())
    records = {row["request_id"]: row for row in original["responses"]}
    reused = json.loads((tmp_path / "valid/results.json").read_text())["responses"]
    assert len(reused) == 423
    assert all(row == records[row["request_id"]] for row in reused)
    original["responses"][0]["body_sha256"] = "changed"
    source = tmp_path / "altered.json"
    study.base.save(source, original)
    with pytest.raises(ValueError, match="mismatch"):
        study.one_shot_subset(manifest, tmp_path / "invalid", source)


def test_luna_changes_only_model_and_reuses_only_available_controls(manifest, tmp_path):
    luna = study.build_manifest("gpt-6-luna")
    assert luna["data"] == manifest["data"]
    assert luna["dataset_sha256"] == manifest["dataset_sha256"]
    assert len(luna["requests"]) == 423
    for a, b in zip(manifest["requests"], luna["requests"], strict=True):
        body = deepcopy(b["body"])
        assert body.pop("model") == "gpt-6-luna"
        assert body == {k: v for k, v in a["body"].items() if k != "model"}
        assert a["body_sha256"] != b["body_sha256"]
    subset = study.one_shot_subset(luna, tmp_path / "luna")
    assert set(subset["cells"]) == {"baseline", "counting_100"}
    rows = json.loads((tmp_path / "luna/results.json").read_text())["responses"]
    assert len(rows) == 282
