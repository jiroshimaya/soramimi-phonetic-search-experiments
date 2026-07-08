# Reproduce Leaderboard Experiments

本家の `leaderboard.md` から外した細かな実験コードと結果を再現するためのスクリプト群です。

## 含めているもの

- `008_*`: gpt-5.4 の prompt / input variation
- `010_*`: gpt-5.4 reasoning effort `medium` の比較
- `011_*`: structured outputs / gpt-5.1 medium の検証
`008_llm_rerank_gpt54.py` と `010_03_llm_rerank_gpt54_medium_step_by_step.py` も比較用の基準として残しています。

## インストール

```bash
uv sync --group dev
```

## 実行例

```bash
cd reproduce_leaderboard

uv run methods/008_01_llm_rerank_gpt54_simple.py
uv run methods/008_02_llm_rerank_gpt54_detailed.py
uv run methods/008_03_llm_rerank_gpt54_step_by_step.py
uv run methods/008_04_llm_rerank_gpt54_detailed_pyopenjtalk_romaji.py
uv run methods/008_04_llm_rerank_gpt54_detailed_pyopenjtalk_romaji_small.py
uv run methods/008_05_llm_rerank_gpt54_detailed_pyopenjtalk_romaji_explicit_small.py
uv run methods/008_06_llm_rerank_gpt54_detailed_kana_and_pyopenjtalk_romaji.py
uv run methods/008_06_llm_rerank_gpt54_detailed_kana_and_pyopenjtalk_romaji_small.py
uv run methods/008_07_llm_rerank_gpt54_nonreasoning_cot_small.py
uv run methods/008_08_llm_rerank_gpt54_detailed_kana_spaced_small.py
uv run methods/008_09_llm_rerank_gpt54_detailed_mora_spaced_small.py
uv run methods/010_01_llm_rerank_gpt54_medium_simple.py
uv run methods/010_02_llm_rerank_gpt54_medium_detailed.py
uv run methods/010_03_llm_rerank_gpt54_medium_step_by_step.py
uv run methods/011_01_probe_cot_structured_outputs.py
uv run methods/011_03_llm_rerank_gpt51_medium_step_by_step.py
```

まとめて流す場合:

```bash
cd reproduce_leaderboard
sh run_all.sh
```

## 結果

- `results/`: full dataset の結果
- `results_small/`: small dataset の結果
- `*_cost_estimate.json`: small run から full dataset へ外挿したコスト試算

`011_03_llm_rerank_gpt51_medium_step_by_step.json` は OpenAI Batch API の実測値を含みます。

## 014: kanasim 変種の leaderboard（LLMなし）

kanasim 0.0.11 で追加された距離オプション（normalize / symmetric / phoneme_unit /
consonant_distance / vowel_binary）の変種を full dataset で評価した結果
（`methods/014_kanasim_variants.py`、詳細は `results/014_kanasim_variants.json`）。

| 設定 | Recall@10 |
|---|--:|
| **baseline: biphone 非対称, vowel_ratio=0.8（leaderboard 掲載値の再現）** | **0.831** |
| normalize, vr=0.9 | 0.832 |
| vowel_binary+normalize, vr=0.3 | 0.826 |
| symmetric, vr=0.8 | 0.821 |
| normalize, vr=0.8 | 0.821 |
| baseline, vr=0.7 | 0.814 |
| vowel_binary+normalize, vr=0.4 | 0.806 |
| baseline, vr=0.6 / vr=0.9 | 0.801 |
| features+normalize, vr=0.8 | 0.799 |
| normalize, vr=0.7 | 0.799 |
| normalize+symmetric, vr=0.8 | 0.796 |
| vowel_binary+normalize, vr=0.5 | 0.796 |
| mono_avg, vr=0.8 | 0.789 |
| vowel_binary+normalize, vr=0.6 | 0.783 |
| vowel_binary+normalize+symmetric, vr=0.8 | 0.773 |
| vowel_binary+normalize, vr=0.8 | 0.769 |
| mono_avg vowel_binary+normalize, vr=0.8 | 0.744 |
| features vowel_binary+normalize, vr=0.8 | 0.744 |
| normalize, vr=0.6 | 0.732 |
| features+normalize, vr=0.6 | 0.672 |

読み取り:

- 新変種はどれも既存の 0.831 を明確には超えない（normalize vr0.9 の 0.832 は誤差の範囲）
- normalize / vowel_binary を使う場合は最適 vowel_ratio が生スケールと大きく変わる
  （normalize は高め 0.9、vowel_binary は低め 0.3 が最適）。オプションと重みはセットで調整が必要
- 母音の 0/1 化（vowel_binary）はチューニングしても連続母音距離に届かず、
  母音の段階的な近さ（ア-オ < ア-イ）がこのタスクの識別に効いていることを示唆
- 対称化（symmetric）はこのタスクでは微減。歌唱ASR誤りマッチング（songasr 側の
  gold 653 評価）では逆に改善しており、タスク依存

## 注意

- dataset 本体と評価関数は `soramimi-phonetic-search-dataset` 依存です
- `openai_batch` backend を使う実験では request JSONL や state JSON が追加生成されます
- API キーは `OPENAI_API_KEY` / `GEMINI_API_KEY` などを使って設定してください
