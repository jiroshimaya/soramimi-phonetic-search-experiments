# Single-period filler on Terra: arithmetic improves, phonetic reranking does not

With 100 space-separated periods, GPT-6.1 Terra improved from 82.33% to 93.17%
on the public arithmetic task. Phonetic macro Recall@10 changed from 80.33% to
80.22%, with no confirmed improvement. This matches the qualitative pattern of
the earlier counting-filler comparison.

[Protocol fixed before new answers](protocol.md) · [Earlier counting study](../018_luna_math_transfer/README.md)

## Results

One first answer per case and condition; `gpt-6.1-terra`, temperature 0,
`reasoning_effort=none`. All reported reasoning-token counts are zero. The
intervals below are paired 95% case-bootstrap intervals for differences in
percentage points (20,000 draws; seed 20261030). They are unadjusted exploratory
intervals and do not measure repeated-generation variability.

| Task | Cases | No filler | Counting 1–100 | Single periods ×100 | Periods vs none (pp) | 95% interval (pp) | Improved / worsened / unchanged |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gen-Arithmetic accuracy | 600 | 82.33% | 95.00% | 93.17% | +10.83 | [+8.00, +13.67] | 74 / 9 / 517 |
| Phonetic macro Recall@10 | 150 | 80.33% | 81.22% | 80.22% | −0.11 | [−4.00, +3.89] | 6 / 6 / 138 |

Arithmetic has 559/600 correct answers with periods versus 494/600 without.
The exact two-sided McNemar p-value is 7.7072e-14. All three conditions have
zero invalid outputs in both tasks. The phonetic interval includes both
decreases and increases; it does not establish a zero effect.

## Filler construction and scope

The new filler consists of 100 ASCII `.` characters joined by single ASCII
spaces: `. . .`. Its exact contents are identical between arithmetic and
phonetic tasks and are inserted before the answer cue in every demonstration
and target. This is distinct from concatenated periods and from the earlier
space-separated `...` condition. Item counts do not imply equal tokenizer-token
counts across filler types.

The construction follows section 3 of
[Brauer et al.](https://arxiv.org/html/2607.03502v2). All other evaluation choices
follow experiment 018: the public Greenblatt Gen-Arithmetic task with ten
demonstrations and a 100-token output limit; phonetic reranking with one
demonstration, detailed instruction, a JSON ranking schema, and a 512-token
output limit. This is a filler-construction follow-up, not a reproduction of
Brauer et al.'s tasks, models, five-shot procedure, or hidden-state analysis.

Arithmetic therefore contains eleven copies of the filler and phonetic two.
Example count, instructions, output format, and total filler amount differ
between tasks, so the task identity alone cannot be isolated as the cause of
the different outcomes. Nor does this comparison establish that counting is
better than periods: filler type and actual token count both differ.

The length was fixed at 100 before collecting period answers to match the
earlier counting condition's item count. This follow-up was selected after
observing the counting results. Baseline/counting answers were collected on
2026-10-30 and reused; period answers were collected on 2026-10-31. Matching
request hashes preserve the specified inputs but cannot exclude provider-side
temporal variation.

## Saved answers and API usage

There are 750 new first answers: 600 arithmetic and 150 phonetic. All 1,500
baseline/counting answers are reused unchanged from experiment 018 after
checking model, dataset, and full request hashes. `results.json` in each task
directory contains both the reused and new first answers; `reuse.json` records
the source-export hash and imported IDs. The original result files are unchanged.
Every response is saved before scoring; wrong/invalid answers are not regenerated.

Provider-reported mean input tokens per request, including demonstrations:

| Task | No filler (reused) | Counting (reused) | Periods (new) |
|---|---:|---:|---:|
| Arithmetic | 997.2 | 3,582.2 | 2,405.2 |
| Phonetic | 924.1 | 1,332.1 | 1,130.1 |

Only new period requests used additional API calls in this follow-up. Their
reported usage totals 1,612,636 input and 8,031 output tokens (1,620,667 total).
The per-condition summaries separate cached-input and cache-write token fields;
raw token totals are not a billing estimate. Latency and a normal CoT comparison
were not evaluated.

## Reproduce or replay

```bash
uv sync --group dev

# Run only missing conditions; exact matching baseline/counting answers are imported.
uv run reproduce_leaderboard/methods/018_filler_math_transfer.py \
  --model gpt-6.1-terra --task math \
  --conditions baseline counting_100 dots_100 --output /tmp/terra-dots-math \
  --reuse reproduce_leaderboard/results/018_luna_math_transfer/terra_math/results.json
uv run reproduce_leaderboard/methods/018_filler_math_transfer.py \
  --model gpt-6.1-terra --task phonetic \
  --conditions baseline counting_100 dots_100 --output /tmp/terra-dots-phonetic \
  --reuse reproduce_leaderboard/results/018_luna_math_transfer/terra_phonetic/results.json

# Offline replay; no API calls.
uv run reproduce_leaderboard/methods/018_filler_math_transfer.py \
  --model gpt-6.1-terra --task math \
  --conditions baseline counting_100 dots_100 --output /tmp/terra-dots-math-replay \
  --replay reproduce_leaderboard/results/019_terra_dot_transfer/math/results.json
uv run reproduce_leaderboard/methods/018_filler_math_transfer.py \
  --model gpt-6.1-terra --task phonetic \
  --conditions baseline counting_100 dots_100 --output /tmp/terra-dots-phonetic-replay \
  --replay reproduce_leaderboard/results/019_terra_dot_transfer/phonetic/results.json
```

Independent rescoring agrees with every saved per-case score in both tasks.
Offline replay reproduces the summaries. New requests require `OPENAI_API_KEY`;
`--prepare-only` inspects the protocol without contacting the model.
