from analytics import kana_edit_distance_nonreasoning as module


def test_kana_edit_distance_handles_basic_cases():
    assert module.kana_edit_distance("アイ", "アイ") == 0
    assert module.kana_edit_distance("アイ", "アオ") == 1
    assert module.kana_edit_distance("アウマル", "アウマ") == 1
    assert module.kana_edit_distance("ショウ", "ショー") == 1


def test_compute_metrics_works():
    metrics = module.compute_metrics(
        [
            {
                "exact_distance": 0,
                "predicted_distance": 0,
                "absolute_error": 0,
                "is_exact_match": True,
            },
            {
                "exact_distance": 1,
                "predicted_distance": 2,
                "absolute_error": 1,
                "is_exact_match": False,
            },
        ]
    )

    assert metrics["pair_count"] == 2
    assert metrics["exact_match_rate"] == 0.5
    assert metrics["mean_absolute_error"] == 0.5


def test_build_user_prompt_supports_spaced_mode():
    class Pair:
        query = "リャオ"
        candidate = "リヤオ"

    prompt = module.build_user_prompt(Pair(), "kana_spaced")

    assert "文字列A: リ ャ オ" in prompt
    assert "文字列B: リ ヤ オ" in prompt


def test_default_output_path_name():
    path = module.build_default_output_path(
        model_name="gpt-5.4",
        sample_size=100,
        seed=7,
        prompt_mode="kana_only",
    )
    assert path.name == "kana_edit_distance_nonreasoning_gpt-5.4_small_sample100_seed7.json"

    spaced_path = module.build_default_output_path(
        model_name="gpt-5.4",
        sample_size=100,
        seed=7,
        prompt_mode="kana_spaced",
    )
    assert (
        spaced_path.name
        == "kana_edit_distance_nonreasoning_gpt-5.4_promptkana_spaced_small_sample100_seed7.json"
    )
