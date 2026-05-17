# soramimi-phonetic-search-experiments

`soramimi-phonetic-search-dataset` から切り出した、leaderboard の細かな実験コードと結果を置くためのリポジトリです。

本家リポジトリには代表的な結果だけを残し、このリポジトリでは次のような派生実験を継続的に管理します。

- 008 系の prompt/input 変種
- 010 系の reasoning medium 比較
- 011 系の probe / gpt-5.1 検証

## 方針

- dataset 本体・評価関数は `soramimi-phonetic-search-dataset` に依存します
- このリポジトリには、細かな実験スクリプト・結果 JSON・補助コードだけを置きます
- 依存先の dataset リポジトリは、切り出し元の commit `6072e13bed37dd2f8eb780e61b6154d21cff2e31` に pin しています

## セットアップ

```bash
uv sync --group dev
```

OpenAI や Gemini を使う実験を再現する場合は、必要な API キーを環境変数に設定してください。

## 構成

- `reproduce_leaderboard/methods/`: 実験スクリプト
- `reproduce_leaderboard/results/`: full dataset の結果
- `reproduce_leaderboard/results_small/`: small dataset の結果
- `reproduce_leaderboard/methods/common/`: 実験用の共通補助コード

詳細な実行方法は [`reproduce_leaderboard/README.md`](reproduce_leaderboard/README.md) にまとめています。

