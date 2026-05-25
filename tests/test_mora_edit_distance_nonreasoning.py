from analytics import mora_edit_distance_nonreasoning as module


def test_load_small_dataset_for_llm_returns_expected_shape():
    dataset = module.load_small_dataset_for_llm(wordlist_size=100)

    assert len(dataset.queries) == 10
    assert all(len(query.wordlist) == 100 for query in dataset.queries)


def test_mora_conversion_and_distance_handle_basic_cases():
    assert module.to_moras("アイ") == ["ア", "イ"]
    assert module.to_moras("キャン") == ["キャ", "ン"]
    assert module.mora_edit_distance("アイ", "アイ") == 0
    assert module.mora_edit_distance("アイ", "アオ") == 1
    assert module.mora_edit_distance("ショウ", "ショー") == 1
    assert module.mora_edit_distance("ア", "") == 1


def test_build_mora_distance_pair_populates_mora_fields():
    pair = module.build_mora_distance_pair(
        pair_id="demo",
        query="アイ",
        candidate="アオ",
    )

    assert pair.query_moras == ["ア", "イ"]
    assert pair.candidate_moras == ["ア", "オ"]
    assert pair.exact_distance == 1


def test_build_prompts_change_by_mode():
    pair = module.build_mora_distance_pair(
        pair_id="demo",
        query="アイ",
        candidate="アオ",
    )

    mora_prompt = module.build_user_prompt(pair, "mora_spaced")
    kana_prompt = module.build_user_prompt(pair, "kana_only")
    kana_hint_prompt = module.build_user_prompt(pair, "kana_only_with_mora_hint")

    assert "モーラ列A: ア イ" in mora_prompt
    assert "モーラ列B: ア オ" in mora_prompt
    assert "モーラ列A:" not in kana_prompt
    assert "文字列A: アイ" in kana_prompt
    assert "文字列B: アオ" in kana_prompt
    assert kana_prompt == kana_hint_prompt


def test_build_system_prompt_adds_simple_mora_hint():
    prompt = module.build_system_prompt("kana_only_with_mora_hint")

    assert "ン / ー / ッ はそれぞれ 1 モーラ" in prompt
    assert "ジャ や シュ のような拗音も 1 モーラ" in prompt


def test_build_default_output_path_adds_prompt_suffix_for_nondefault_mode():
    default_path = module.build_default_output_path(
        model_name="gpt-5.4",
        sample_size=100,
        seed=7,
        prompt_mode="mora_spaced",
    )
    kana_only_path = module.build_default_output_path(
        model_name="gpt-5.4",
        sample_size=100,
        seed=7,
        prompt_mode="kana_only",
    )
    kana_hint_path = module.build_default_output_path(
        model_name="gpt-5.4",
        sample_size=100,
        seed=7,
        prompt_mode="kana_only_with_mora_hint",
    )

    assert default_path.name == "mora_edit_distance_nonreasoning_gpt-5.4_small_sample100_seed7.json"
    assert (
        kana_only_path.name
        == "mora_edit_distance_nonreasoning_gpt-5.4_promptkana_only_small_sample100_seed7.json"
    )
    assert (
        kana_hint_path.name
        == "mora_edit_distance_nonreasoning_gpt-5.4_promptkana_only_with_mora_hint_small_sample100_seed7.json"
    )


def test_build_all_pairs_and_metrics_work():
    pairs = module.build_all_pairs(wordlist_size=100)
    sampled_pairs = module.select_pairs(pairs, sample_size=3, seed=7)

    assert len(pairs) == 1000
    assert len(sampled_pairs) == 3
    assert len({pair.pair_id for pair in sampled_pairs}) == 3

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
            {
                "exact_distance": 2,
                "predicted_distance": 2,
                "absolute_error": 0,
                "is_exact_match": True,
            },
        ]
    )

    assert metrics["pair_count"] == 3
    assert metrics["exact_match_rate"] == 2 / 3
    assert metrics["mean_absolute_error"] == 1 / 3
    assert metrics["pearson_correlation"] == 0.8660254037844385
