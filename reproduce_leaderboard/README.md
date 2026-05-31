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

## 注意

- dataset 本体と評価関数は `soramimi-phonetic-search-dataset` 依存です
- `openai_batch` backend を使う実験では request JSONL や state JSON が追加生成されます
- API キーは `OPENAI_API_KEY` / `GEMINI_API_KEY` などを使って設定してください
