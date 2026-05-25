from analytics import mora_segmentation_nonreasoning as module


def test_collect_unique_strings_and_sampling_work():
    samples = module.collect_unique_strings(wordlist_size=100)
    sampled = module.select_samples(samples, sample_size=100, seed=7)

    assert len(samples) >= 100
    assert len({sample.text for sample in samples}) == len(samples)
    assert len(sampled) == 100


def test_build_system_prompt_mentions_simple_mora_rules():
    prompt = module.build_system_prompt()

    assert "ン / ー / ッ はそれぞれ 1 モーラ" in prompt
    assert "ジャ や シュ のような拗音も 1 モーラ" in prompt


def test_compute_metrics_works():
    metrics = module.compute_metrics(
        [
            {
                "is_exact_match": True,
                "mora_count_match": True,
                "gold_mora_count": 2,
                "predicted_mora_count": 2,
            },
            {
                "is_exact_match": False,
                "mora_count_match": True,
                "gold_mora_count": 3,
                "predicted_mora_count": 3,
            },
            {
                "is_exact_match": False,
                "mora_count_match": False,
                "gold_mora_count": 4,
                "predicted_mora_count": 3,
            },
        ]
    )

    assert metrics["sample_count"] == 3
    assert metrics["exact_match_rate"] == 1 / 3
    assert metrics["mora_count_match_rate"] == 2 / 3
    assert metrics["mean_absolute_mora_count_error"] == 1 / 3
