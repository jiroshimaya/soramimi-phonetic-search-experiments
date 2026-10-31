# Luna and Terra on the same ten-shot phonetic evaluation

Both models now use exactly the same ten demonstrations, 141 held-out queries,
100 candidates per query, and three filler conditions. Ten demonstrations plus
the target give eleven copies of the filler, matching arithmetic's copy count.
Only the model ID differs between their new phonetic request bodies.

[Protocol fixed before Luna answers](protocol.md) · [Terra's unchanged results and demonstration construction](../020_terra_phonetic_ten_shot/README.md)

## Results

Macro Recall@10, temperature 0, `reasoning_effort=none`, one first answer per
query and condition. All new reported reasoning-token counts are zero.

| Model | No filler | Counting 1–100 | Single periods ×100 |
|---|---:|---:|---:|
| GPT-6 Luna | 52.78% | 52.78% | 54.37% |
| GPT-6.1 Terra (reused) | 79.55% | 80.79% | 82.62% |

| Model | Filler | Gain vs none (pp) | Paired 95% interval (pp) | Improved / worsened / unchanged |
|---|---|---:|---:|---:|
| GPT-6 Luna | Counting | 0.00 | [−4.91, +4.91] | 9 / 8 / 124 |
| GPT-6 Luna | Periods | +1.60 | [−2.84, +6.21] | 8 / 5 / 128 |
| GPT-6.1 Terra | Counting | +1.24 | [−2.30, +4.96] | 5 / 4 / 132 |
| GPT-6.1 Terra | Periods | +3.07 | [−0.83, +7.21] | 8 / 3 / 130 |

All conditions have zero invalid outputs. No filler comparison confirms an
improvement. Equal mean scores can still have unequal improved/worsened counts
because each query's recall can change by a different fraction.

Intervals are unadjusted exploratory paired case-bootstrap intervals with
20,000 draws and seed 20261030. They do not establish zero effect or absence of
a shot-count effect, and do not represent repeated-generation variability.

## Reused controls and scope

There are 423 new Luna first responses. All Terra and arithmetic results are
reused unchanged; no requests for those controls were sent. Luna's existing
one-shot baseline and counting responses are rescored on the same 141 queries
(282 unchanged saved answers) after checking their original request hashes.
The matched one-shot means are 52.48% without filler and 56.68% with counting.

No one-shot single-period-100 Luna result exists. The earlier `...`-300
condition is a different construction and is not used as a substitute. The
missing comparison is explicit in `comparison.json`; no extra one-shot calls
are made. No prior result files are changed.

The nine corpus demonstrations and original synthetic example are exactly
those used for Terra. Demonstration queries are excluded from evaluation. New
example answers show annotated positives with Top N=1/2 rather than invented
top-ten rankings. The original example retains Top N=5. Evaluation asks for ten
indices. See experiment 020 for the fixed indices, seed, prompt format and
remaining differences from arithmetic. Equal shot/copy counts do not equalize
total input tokens, instructions, message roles or output format.

The Luna follow-up was selected after observing Terra. Historical controls are
from earlier days; new responses were collected on 2026-11-01. Model selection,
example content, and provider-side temporal variation limit causal claims.

## Saved answers, usage and validation

`results.json` contains the frozen protocol and all 423 new first answers;
`summary.json` contains per-query scores and filler contrasts. `one_shot_subset/`
contains the 282 reused responses and their matched summary.
`comparison.json` records shot-count contrasts where a matching control exists.

Every response is saved before scoring. Invalid or wrong answers are not
regenerated. Independent JSON/index-based scoring matches all new per-query
scores. Offline replay reproduces the results; Terra's existing replay is
unchanged, including request hashes and summaries.

Provider-reported new Luna usage totals 3,204,411 input tokens and 14,382 output
tokens (3,218,793 total). Cached-input tokens total 2,861,524 and cache-write
tokens 341,618; categories are recorded separately and raw totals are not a
billing estimate. Mean input tokens per request are 6,449.8 without filler,
8,693.8 with counting and 7,582.8 with periods. No latency comparison is made.

```bash
uv sync --group dev

# Prepare without API calls, then run only missing Luna requests.
uv run reproduce_leaderboard/methods/020_terra_phonetic_ten_shot.py \
  --model gpt-6-luna --output /tmp/luna-ten-shot --prepare-only
uv run reproduce_leaderboard/methods/020_terra_phonetic_ten_shot.py \
  --model gpt-6-luna --output /tmp/luna-ten-shot --concurrency 8

# Fully offline replay; no API calls.
uv run reproduce_leaderboard/methods/020_terra_phonetic_ten_shot.py \
  --model gpt-6-luna --output /tmp/luna-ten-shot-replay \
  --replay reproduce_leaderboard/results/021_luna_phonetic_ten_shot/results.json
```

New requests require `OPENAI_API_KEY`. The script still defaults to Terra for
backward-compatible replay of experiment 020. A changed manifest cannot reuse
an existing output directory's first answers.
