# Terra single-period filler follow-up

Specified on 2026-10-31 before collecting any single-period answers. This is a
follow-up after observing the counting-filler results, not a new independent
replication of the baseline. No model or length selection uses dot-condition
answers.

- Model: `gpt-6.1-terra` only; `reasoning_effort=none`, temperature 0.
- New filler: 100 ASCII periods joined by single ASCII spaces (`. . .`), with no
  leading/trailing space in the filler itself. It is neither 100 concatenated
  periods nor 100 repetitions of `...`.
- The construction follows [Brauer et al.](https://arxiv.org/html/2607.03502v2),
  section 3. Its length is fixed at 100 to match the existing counting-100 item
  count. Actual tokenizer-token counts need not match between types.
- Arithmetic: the same 600 Gen-Arithmetic cases, ten demonstrations, numerical
  parser, and 100-token output limit as experiment 018. Extend the original
  instruction to describe "100 dots" in place of "counting from 1 to 100".
- Phonetic: the same 150 queries, fixed 100 candidates, detailed instruction,
  one demonstration, JSON schema, and 512-token output limit as experiment 018.
- Insert the exact same filler after each problem and before the answer cue in
  both tasks, including their demonstrations. Arithmetic still has ten examples
  and phonetic one, so total filler/input-token counts differ between tasks.
- Reuse all baseline and counting-100 first answers from experiment 018 only
  after verifying dataset, model, and complete request hashes. No new baseline
  or counting requests. Only 600 + 150 dot requests are authorized by this plan.
- Persist every first answer before scoring, including invalid answers. Do not
  regenerate invalid or wrong answers. Resume only missing requests.
- Compare dots-100 against baseline with unchanged metrics: arithmetic exact
  accuracy and phonetic macro Recall@10. Report paired 20,000-draw bootstrap
  95% intervals (seed 20261030), invalid counts, and improved/worsened cases.
  Retain the original counting comparison alongside the follow-up. Intervals
  are unadjusted and the follow-up is exploratory.
- Keep all old result files unchanged. Record imported-response provenance and
  report new API usage separately from the usage of reused responses.

This follows the paper's filler *construction* on the existing public arithmetic
and phonetic tasks. It does not reproduce its tasks, five-shot prompts, models,
or hidden-state analyses. Baselines were collected on 2026-10-30; dot responses
are collected later, so request-hash equality does not remove possible
provider-side temporal variation. One first generation is used per condition.
