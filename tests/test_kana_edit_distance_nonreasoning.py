from reproduce_leaderboard.ablations import mora_edit_distance_nonreasoning as module


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
