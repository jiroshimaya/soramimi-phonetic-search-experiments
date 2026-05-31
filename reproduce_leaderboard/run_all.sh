#!/bin/bash

set -euo pipefail

echo "=== 008 baseline ==="
uv run methods/008_llm_rerank_gpt54.py

echo "=== 008 prompt/input variants ==="
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

echo "=== 010 reasoning medium variants ==="
uv run methods/010_01_llm_rerank_gpt54_medium_simple.py
uv run methods/010_02_llm_rerank_gpt54_medium_detailed.py
uv run methods/010_03_llm_rerank_gpt54_medium_step_by_step.py

echo "=== 011 probes ==="
uv run methods/011_01_probe_cot_structured_outputs.py
uv run methods/011_03_llm_rerank_gpt51_medium_step_by_step.py

echo "Done!"
