# Ten-shot phonetic reranking with Terra

With ten demonstrations, GPT-6.1 Terra's phonetic macro Recall@10 was 79.55%
without filler, 80.79% with counting 1–100, and 82.62% with 100 space-separated
periods. Neither filler gain was confirmed by the paired 95% interval. The
period gain was +3.07 percentage points, with an interval of [−0.83, +7.21].

[Protocol fixed before new responses](protocol.md) · [Earlier arithmetic and one-shot results](../019_terra_dot_transfer/README.md)

## Results on the same 141 evaluation queries

Nine queries were held out for demonstrations. Both one-shot and ten-shot
results below use exactly the remaining 141 queries and their original fixed
100-candidate lists. The one-shot numbers are rescored saved responses; they
are not the earlier all-150-query averages.

| Examples | Filler | No-filler Recall@10 | With-filler Recall@10 | Gain (pp) | Paired 95% interval (pp) | Improved / worsened / unchanged |
|---|---|---:|---:|---:|---:|---:|
| 1 | Counting 1–100 | 80.50% | 82.15% | +1.65 | [−1.77, +5.32] | 6 / 3 / 132 |
| 1 | Single periods ×100 | 80.50% | 80.38% | −0.12 | [−4.26, +4.02] | 6 / 6 / 129 |
| 10 | Counting 1–100 | 79.55% | 80.79% | +1.24 | [−2.30, +4.96] | 5 / 4 / 132 |
| 10 | Single periods ×100 | 79.55% | 82.62% | +3.07 | [−0.83, +7.21] | 8 / 3 / 130 |

All six cells have zero invalid outputs. The intervals are unadjusted
exploratory paired case-bootstrap intervals: 20,000 draws, seed 20261030. They
measure case-sampling uncertainty, not repeated-generation variability.

To assess whether the *filler gain* changed, subtract the one-shot gain from the
ten-shot gain on each query and bootstrap that paired difference:

| Filler | Change in gain, 10-shot minus 1-shot (pp) | 95% interval (pp) |
|---|---:|---:|
| Counting 1–100 | −0.41 | [−5.08, +4.26] |
| Single periods ×100 | +3.19 | [−2.36, +9.22] |

Neither change is confirmed. These results do not establish that example count
has no effect, nor that the filler effect is zero. In particular, the ten-shot
period interval still allows a positive gain of several percentage points.

## Demonstrations and alignment with arithmetic

The original synthetic example is preserved exactly. Nine additional examples
are chosen reproducibly from the pinned phonetic corpus with seed 20261101,
without looking at model scores. Their queries occur only once in the corpus;
all nine queries are excluded from evaluation. Their fixed source indices are
25, 31, 102, 17, 13, 14, 58, 72 and 86.

Each new demonstration shows its original 100 candidates and the annotated
positive indices. Its Top N equals its positive count: one for eight examples,
two for one example. Unannotated ranks are not supplied as invented top-ten
truth. The original synthetic example still has eight candidates and Top N=5.
Evaluation requests still ask for ten distinct candidate indices.

The existing system-message prefix, instructions, original example, target
user message, candidate order, response schema and output limit are preserved.
Only the nine examples are appended to the system message. All conditions use
the same examples in the same order, with filler before every example answer
and the target answer.

Arithmetic and phonetic reranking now both have ten demonstrations and eleven
filler copies per request. The filler strings and their 100-item lengths match
the existing arithmetic conditions. Arithmetic's saved results remain 82.33%
without filler, 95.00% with counting, and 93.17% with periods; no arithmetic
requests were repeated.

This aligns example count and filler-copy count, not the entire experimental
setting. Total input tokens, instructions, answer format, message roles and
demonstration supervision still differ between tasks. Adding examples also
changes their content, including in-domain gold labels, so the shot comparison
does not isolate a universal causal effect of example count alone. Controls
were collected on 2026-10-30/31 and new responses on 2026-11-01; provider-side
temporal variation remains possible.

## Saved data, validation and input usage

`results.json` contains all 423 new first responses and the frozen data and
protocol. `summary.json` contains their per-query scores and paired contrasts.
`one_shot_subset/` contains the 423 unchanged saved responses used for the
matched comparison, their source-export hash, and rescored summaries.
`comparison.json` records ten-minus-one-shot contrasts and changes in filler
gains. All previous result files are unchanged.

Independent JSON/index-based rescoring matches every new per-query score.
Offline replay reproduces the new, matched-control and interaction summaries.
All new provider-reported reasoning-token counts are zero. Responses are
saved before scoring; invalid or wrong answers are not regenerated.

| Condition | Mean input tokens, 1-shot (reused) | Mean input tokens, 10-shot (new) |
|---|---:|---:|
| No filler | 924.8 | 6,449.8 |
| Counting 1–100 | 1,332.8 | 8,693.8 |
| Single periods ×100 | 1,130.8 | 7,582.8 |

Only the 423 ten-shot responses required new API calls. Their total reported
usage is 3,204,411 input and 14,382 output tokens (3,218,793 total). Cached-input
and cache-write fields are recorded separately in summaries; raw token totals
are not a billing estimate. No latency comparison was performed.

## Run or replay

```bash
uv sync --group dev

# Inspect the frozen examples and 423 requests without API calls.
uv run reproduce_leaderboard/methods/020_terra_phonetic_ten_shot.py \
  --output /tmp/terra-ten-shot --prepare-only

# Collect only missing ten-shot responses; rescore saved one-shot answers offline.
uv run reproduce_leaderboard/methods/020_terra_phonetic_ten_shot.py \
  --output /tmp/terra-ten-shot --concurrency 8

# Fully offline: no API calls.
uv run reproduce_leaderboard/methods/020_terra_phonetic_ten_shot.py \
  --output /tmp/terra-ten-shot-replay \
  --replay reproduce_leaderboard/results/020_terra_phonetic_ten_shot/results.json
```

New requests require `OPENAI_API_KEY`. Reusing the same output directory resumes
saved first responses and rejects a changed manifest. Existing one-shot exports
are verified against their complete original request hashes before rescoring;
they are never used as answers to the different ten-shot requests.
