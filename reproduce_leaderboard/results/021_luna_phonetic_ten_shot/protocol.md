# Luna ten-shot phonetic follow-up

Fixed before collecting new responses on 2026-11-01. Use exactly experiment
020's demonstrations, held-out indices, 141 evaluation queries, candidate order,
JSON schema, 512-token output limit and shuffled schedule. Change only the model
to `gpt-6-luna`, keeping `reasoning_effort=none` and temperature 0.

Evaluate all three conditions: no filler, counting 1–100, and 100 space-separated
single periods. Ten demonstrations plus the target give eleven filler copies.
Generate 423 new first responses. Save every response before scoring, count
invalid answers as zero, and never regenerate an incorrect or invalid answer.
Resume only missing requests after service/transport failures.

Reuse all Terra ten-shot results unchanged; make no new Terra or arithmetic
requests. Rescore Luna's existing one-shot baseline/counting responses on the
same 141 queries (282 saved responses), verifying their original request hashes.
No Luna one-shot single-period-100 condition exists. Do not substitute the
earlier `...`-300 condition or generate a new one-shot period control.

Report macro Recall@10 and paired 95% case-bootstrap intervals (20,000 draws,
seed 20261030), including invalid counts and improved/worsened cases. Primary
follow-up contrast: periods versus none; counting is also reported. Compare
shots only where matching saved controls exist. All intervals are unadjusted
exploratory intervals. Keep all existing results unchanged.

The construction and limitations of experiment 020 still apply: nine corpus
queries are excluded for demonstrations, example answers show annotated
positives with Top N=1/2, and adding examples also changes content. Arithmetic
and phonetic tasks match in shot count and filler-copy count, not total input
tokens, instructions, message roles or output format. Historical controls and
new responses are collected at different times. This follow-up was selected
after observing Terra's results.
