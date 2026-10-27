# Filler tokens in phonetic reranking

Adding filler did not improve Recall@10 in this GPT-5.4 experiment. Space-separated
dots reduced recall; counting from 1 to 300 had a smaller negative point estimate
whose exploratory interval includes zero.

## Results

All values average three repetitions of the same 150 queries. Intervals are
paired, exploratory 95% bootstrap intervals for the change in percentage points.
The primary comparison uses the existing reranker's structured-output schema.

| Filler | Recall@10 | Change (pp) | 95% interval (pp) | Invalid / 450 |
|---|---:|---:|---:|---:|
| None | 57.22% | — | — | 1 |
| 300 consecutive dots | 56.56% | -0.67 | [-4.11, +2.78] | 0 |
| 300 space-separated dots | 52.96% | -4.26 | [-7.63, -1.04] | 0 |
| Counting 1 through 300 | 55.35% | -1.87 | [-5.33, +1.50] | 0 |

The counting comparison reuses the exact 450 structured baseline responses;
only its 450 treatment responses were newly generated. Independent repetitions
keep separate cache slots even when their prompt hashes are identical.

The secondary plain-output experiment returned Recall@10 of 54.63% without
filler, 54.89% with consecutive dots, and 54.31% with spaced dots. Its respective
invalid-output counts were 0, 11, and 1 out of 450. Both paired intervals include
zero; these scores include zero credit for malformed rankings.

## Relation to the sources

[Greenblatt's experiment](https://www.lesswrong.com/posts/NYzYJ2WoB74E6uj9L), cited by
[joisino](https://joisino.hatenablog.com/entry/filler), primarily uses counting
from 1 through N. Its reported 45% to 51% result uses N=300, not necessarily 300
model tokens. Our counting string is 599 `o200k_base` tokens.

[Brauer et al., section 3](https://arxiv.org/html/2607.03502v2) use space-separated
dots, counting, or letters between the question and answer cue. Their length k
counts filler items, not tokenizer tokens. Our spaced-dot string has the same
construction as the dots in their [published code](https://github.com/kaleybrauer/filler-token-reasoning/blob/main/scripts/prompt_utils.py).
Consecutive dots are an additional control: 300 characters encode as only 6
`o200k_base` tokens, whereas the 300 spaced dots encode as 300 tokens. Counts are
for filler text alone, excluding its newline.

This is a transfer to phonetic reranking, not a reproduction of the original
math/fact tasks. We retain the existing `detailed` instruction and single example,
without filler in that example. Brauer et al. use five examples containing filler
and DeepSeek V3/Kimi K2; Greenblatt's highlighted result uses Claude Opus 4.5.
We use GPT-5.4 and structured output for the primary comparison. These results do
not rule out gains with the originals' models, complete prompts, or other tasks.

## Protocol

- Dataset commit: `6072e13bed37dd2f8eb780e61b6154d21cff2e31`.
- All 150 queries, with the same ordered 100 candidates in every condition.
  Candidates contain all positives plus hard negatives. This measures conditional
  reranking, not end-to-end retrieval over the lexicon.
- Returned model: `gpt-5.4-2026-03-05`; `reasoning_effort=none`, default sampling
  parameters, `max_completion_tokens=512`, no reasoning output.
- Insert filler immediately before the user's `Reranked:` cue; baseline unchanged.
- Three repetitions per query and condition. Shuffle scheduling with seed
  20261027; concurrency 8. The counting extension runs after the dot comparison.
- Save each first answer. Require ten distinct candidate indices; invalid or
  incomplete rankings score zero and are never retried. Only transient
  HTTP/transport failures are retried.
- Macro Recall@10. Average repetitions within query, then bootstrap queries with
  20,000 draws and seed 20261027. The exploratory intervals are not adjusted for
  multiple comparisons; they do not establish internal model computation.

## Reproduce

From the repository root:

```bash
uv sync --group dev
# Prepare without API calls; remove --prepare-only to execute:
uv run reproduce_leaderboard/methods/017_filler_tokens.py \
  --structured --output /tmp/filler-new --prepare-only
# Add only the counting condition, reusing matching baseline/replicate slots:
uv run reproduce_leaderboard/methods/017_filler_tokens.py \
  --structured --counting --reuse-output /tmp/filler-new --output /tmp/filler-counting
# Recompute the published primary results without API calls:
uv run reproduce_leaderboard/methods/017_filler_tokens.py \
  --structured --output /tmp/filler-replay \
  --replay reproduce_leaderboard/results/017_filler_tokens/structured/results.json
```

Omit `--structured` and use `plain/results.json` to replay the plain-output run.
Add `--counting` and use `counting/results.json` to replay the counting extension.
Executing new requests requires `OPENAI_API_KEY`; repeating a command resumes
its saved responses. Changed prompt/model/settings hashes cannot reuse a cache
entry. Replaying or summarizing results makes no API calls.

Each directory contains `results.json` (fixed candidates, prompts, original and
parsed answers, request hashes, model version, token usage) and `summary.json`
(repetition recalls, per-query scores, intervals). Baseline records in `counting`
are copies of those in `structured`, not additional independent observations.
