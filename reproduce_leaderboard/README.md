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

## 5種類の言い換えを含むモデル比較

`016_paper_model_comparison.py` は、GPT-5.6の3モデルについて、指示文3種類（simple / detailed / step_by_step）、reasoning effort 2種類（none / medium）、言い換え5種類（v1–v5）の全90試行を準備します。各試行は固定の150クエリ・100候補から上位10件を返します。既存の008・010・012・013系と同じ入力文・候補・評価方法を使い、入力変換は行いません。

```bash
# リポジトリ直下で実行。計画の作成だけではAPIを呼び出しません。
uv run reproduce_leaderboard/methods/016_paper_model_comparison.py --output /path/to/manifest.json

# 完了した試行からJSON・Markdown・LaTeXの比較表を作成します。
uv run analytics/paper_model_comparison_stats.py --results-dir /path/to/results --output-dir /path/to/summary
```

reasoning effortは明示的に指定し、出力上限はnoneで1,000、mediumで24,000トークンです。実行は同期APIを使用するため、既存のBatch API実験とは実行方式が異なります。比較表は各条件の5試行が揃ってから、Recall@10の平均と標本標準偏差（ddof=1）を出力します。難易度別の件数はeasy 65・medium 47・hard 38で、全体150件も併記します。未完了の条件はpendingと表示します。

`common/daily_free_reranker.py` は確認済みの日次枠と進捗を読み、完了したリクエストを再送せずに実行を再開します。利用条件の確認が必要なモデルは実行対象に含めません。結果ファイルは150件が完了した試行単位で保存します。

全90試行の結果は [`results/016_paper_model_comparison/`](results/016_paper_model_comparison/) に、難易度別の集計は [`../analytics/results/016_paper_model_comparison/`](../analytics/results/016_paper_model_comparison/) に保存しています。各モデルで最も高かった条件は medium / step_by_step でした。

| Model | Easy (65) | Medium (47) | Hard (38) | Overall |
|---|---:|---:|---:|---:|
| gpt-5.6-luna | 0.960 ± 0.015 | 0.702 ± 0.047 | 0.625 ± 0.041 | 0.794 ± 0.024 |
| gpt-5.6-terra | 0.969 ± 0.011 | 0.787 ± 0.051 | 0.752 ± 0.035 | 0.857 ± 0.010 |
| gpt-5.6-sol | 0.975 ± 0.008 | 0.893 ± 0.032 | 0.862 ± 0.052 | 0.921 ± 0.022 |

値は Recall@10 の平均 ± 標本標準偏差（言い換え5試行）です。全18条件の表は [`paper_model_comparison.md`](../analytics/results/016_paper_model_comparison/paper_model_comparison.md) を参照してください。

### 日次枠に合わせた並列実行

`--concurrency` で同時実行数を指定できます（省略時は1）。例えば、確認済みの利用枠と保存済みの進捗を使い、最大6件を並列に実行するには次のように指定します。

```bash
uv run reproduce_leaderboard/methods/common/daily_free_reranker.py \
  --manifest /path/to/manifest.json \
  --baseline /path/to/baseline.json \
  --checkpoint /path/to/state.json \
  --output-dir /path/to/results \
  --concurrency 6 --max-requests 30
```

実行中のリクエストの最大出力量も含めて日次枠を確保し、残量に収まる分だけ送信します。残量が少なくなると同時実行数も減ります。並列数を変更しても、入力文・候補順・評価条件と完了済みの結果は引き継がれます。個別の出力不備は記録して、ほかの項目を続けます。重複などで上位10件を取得できない場合は未実行の項目を優先し、同じ入力で最大2回再試行します。応答と使用量は試行ごとに保存し、再試行分も日次枠と結果の使用量に含めます。成否が不明なリクエストや再試行上限に達した項目は保留にし、ほかの項目の実行を妨げません。保留を含む試行は完了結果として保存しません。
