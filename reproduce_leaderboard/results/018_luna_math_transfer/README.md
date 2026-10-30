# Filler validation on public arithmetic and transfer to phonetic reranking

On the source arithmetic task, GPT-6.1 Terra improved from 82.33% to 95.00%
with counting filler 1–100. GPT-6 Luna did not show a confirmed gain. Neither
model showed a confirmed phonetic reranking improvement in this evaluation.

[Frozen protocol](protocol.md) · [Japanese article outline](../../../articles/filler_tokens_outline_ja.md)

## Main comparison

One first response per case/condition, temperature 0 and `reasoning_effort=none`.
All reported reasoning-token counts are zero. Intervals are paired 95% case-bootstrap
intervals for the change in percentage points (20,000 draws; seed 20261030).

| Task / cases | Model | No filler | Counting 1–100 | Change (pp) | 95% interval (pp) | Improved / worsened / unchanged |
|---|---|---:|---:|---:|---:|---:|
| Gen-Arithmetic accuracy (600) | GPT-6 Luna | 19.17% | 21.17% | +2.00 | [-1.00, +5.00] | 48 / 36 / 516 |
| Gen-Arithmetic accuracy (600) | GPT-6.1 Terra | 82.33% | 95.00% | +12.67 | [+10.00, +15.50] | 79 / 3 / 518 |
| Phonetic Recall@10 (150) | GPT-6 Luna | 53.33% | 56.61% | +3.28 | [-1.17, +8.00] | 10 / 4 / 136 |
| Phonetic Recall@10 (150) | GPT-6.1 Terra | 80.33% | 81.22% | +0.89 | [-2.78, +4.56] | 6 / 4 / 140 |

The exact two-sided McNemar p-values for the arithmetic comparison are 0.229864
for Luna and 3.80354e-20 for Terra. Terra is a follow-up model selected after the
Luna results; its cases, demonstrations, and baseline/counting-100 conditions
were fixed before collecting Terra answers. Both models and all tested conditions
are reported. The paired intervals do not measure variability across repeated
generations of the same prompt.

The phonetic result does not establish zero effect. Its interval spans both
decreases and increases. It does not yet support changing the default reranker
configuration on the strength of filler alone. Example count, task instructions,
and output format differ between arithmetic and phonetic tasks, so these results
do not isolate task identity as the causal explanation.

## Additional Luna conditions

These conditions are exploratory; the primary arithmetic comparison was counting 100.

| Task | Condition | Score | Change (pp) | 95% interval (pp) | Invalid answers |
|---|---|---:|---:|---:|---:|
| Gen-Arithmetic accuracy (600) | counting_300 | 19.00% | -0.17 | [-3.00, +2.50] | 1 |
| Gen-Arithmetic accuracy (600) | ellipsis_300 | 17.33% | -1.83 | [-4.67, +1.00] | 5 |
| Phonetic Recall@10 (150) | counting_300 | 56.83% | +3.50 | [-1.33, +8.67] | 0 |
| Phonetic Recall@10 (150) | ellipsis_300 | 51.89% | -1.44 | [-5.83, +2.89] | 1 |

## Input tokens and invalid answers

Token counts below are provider-reported input counts averaged over requests.
The full summaries separate cached-input and cache-write tokens; these counts
are not a billing estimate. Request latency and a chain-of-thought comparison
were not measured.

| Evaluation | Mean input, none | Mean input, counting 100 | Additional input / request | Invalid none / counting 100 |
|---|---:|---:|---:|---:|
| GPT-6 Luna, math | 997.2 | 3582.2 | +2585.0 | 5 / 2 |
| GPT-6.1 Terra, math | 997.2 | 3582.2 | +2585.0 | 0 / 0 |
| GPT-6 Luna, phonetic | 924.1 | 1332.1 | +408.0 | 0 / 0 |
| GPT-6.1 Terra, phonetic | 924.1 | 1332.1 | +408.0 | 0 / 0 |

## What is reproduced

The public Gen-Arithmetic task and 10-shot prompt procedure come from
[Greenblatt’s study](https://www.lesswrong.com/posts/NYzYJ2WoB74E6uj9L) and its
[MIT-licensed implementation](https://github.com/rgreenblatt/no_cot_math_public/tree/7f4d3e4e2f775db1ef45986c4996be9d7a34c7e1).
The fixed 600 test expressions are drawn from its 3,000 published expressions;
all 10 demonstrations are excluded. Reference answers are independently checked
with restricted integer AST evaluation. The four prompt conditions were compared
byte-for-byte with the upstream OpenAI prompt helpers.

This is a cross-model behavioral validation. The private competition-math set
behind the cited 45%→51% result is unavailable, so that score is not reproduced.
The hidden-state analyses in [Brauer et al.](https://arxiv.org/html/2607.03502v2)
are also outside scope. Its spaced-dot construction is distinct from this
implementation’s `ellipsis` filler: this study follows the original arithmetic
code’s space-joined ASCII `...`. Filler-item counts are not tokenizer-token counts.

The models are [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
and [GPT-6.1 Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra),
both using their supported `none` reasoning setting. Reported model IDs and
complete provider token usage are retained. Results were collected on 2026-10-30.

Phonetic transfer uses the existing detailed instruction, one example, and JSON
output schema. Filler precedes the answer cue in both example and target. Every
positive is present among 100 candidates, so the metric measures conditional
reranking, not speech recognition or search over the full lexicon.

## Reproduce

```bash
uv sync --group dev
# Add --prepare-only to inspect requests without API calls.
uv run reproduce_leaderboard/methods/018_filler_math_transfer.py \
  --model gpt-6-luna --task math --output /tmp/filler-luna-math
uv run reproduce_leaderboard/methods/018_filler_math_transfer.py \
  --model gpt-6-luna --task phonetic --output /tmp/filler-luna-phonetic
uv run reproduce_leaderboard/methods/018_filler_math_transfer.py \
  --model gpt-6.1-terra --task math --conditions baseline counting_100 \
  --output /tmp/filler-terra-math
uv run reproduce_leaderboard/methods/018_filler_math_transfer.py \
  --model gpt-6.1-terra --task phonetic --conditions baseline counting_100 \
  --output /tmp/filler-terra-phonetic

# Offline replay: no API calls.
uv run reproduce_leaderboard/methods/018_filler_math_transfer.py \
  --model gpt-6.1-terra --task math --conditions baseline counting_100 \
  --output /tmp/filler-replay \
  --replay reproduce_leaderboard/results/018_luna_math_transfer/terra_math/results.json
```

`results.json` holds fixed cases, demonstrations, parameters, original answers,
request hashes, and provider usage; `summary.json` holds scores and paired
comparisons. Replay the other three directories with matching model/task/condition
arguments. Executing new requests requires `OPENAI_API_KEY`. Repeating the same
command resumes saved answers; changing request parameters cannot reuse a cached
response. A successful provider response is saved before scoring and is never
regenerated merely because its answer is invalid. There are 4,500 first responses
in the four published evaluation sets.
