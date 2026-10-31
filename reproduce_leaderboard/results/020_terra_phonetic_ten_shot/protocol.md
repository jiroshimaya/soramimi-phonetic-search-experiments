# Ten-shot phonetic follow-up

Fixed on 2026-11-01 before collecting new responses. The experiment follows the
earlier one-shot results; it is exploratory and does not select examples using
their model scores.

- Model: `gpt-6.1-terra`, `reasoning_effort=none`, temperature 0, 512 maximum
  completion tokens, unchanged JSON ranking schema and detailed instructions.
- Conditions: no filler, counting from 1 to 100, and 100 ASCII periods joined
  by ASCII spaces. Use the same strings as the arithmetic experiment.
- Retain the original synthetic example exactly. Add nine annotated examples
  from the pinned phonetic dataset, using seed 20261101 and `random.sample` on
  source-ordered cases whose query occurs exactly once and whose positive count
  is between one and ten. Fixed source indices, in prompt order:
  **25, 31, 102, 17, 13, 14, 58, 72, 86**.
- Each additional example includes its fixed 100 candidates. Set its requested
  Top N to its number of annotated positives and give those indices as the
  answer, in candidate-index order. Do not invent unannotated top-ten rankings.
  Eight new examples have one positive; one has two. The original synthetic
  example retains its eight candidates and five-index answer.
- Exclude all nine demonstration queries from evaluation. Score the remaining
  141 queries, with their original candidate lists, positives and source IDs.
  Confirm that the original synthetic query is absent from evaluation too.
- Extend the existing system message with the nine examples, preserving its
  one-shot prefix and the complete target user message. Do not change model,
  schema, output limit, candidate order or instruction text. All three
  conditions use the same examples in the same order.
- Insert filler immediately before every example answer and the target answer:
  eleven copies per request, matching arithmetic's ten examples plus target.
  No filler in the no-filler condition.
- New responses: 141 queries × three conditions = 423. Persist every first
  response before scoring; invalid/wrong answers count as zero and are never
  regenerated. Resume only missing requests after service/transport errors.
- Recompute one-shot scores on the same 141 queries from the saved 019 exports,
  validating all source request hashes and protocol metadata. Do not call the
  API for one-shot or arithmetic controls, or import one-shot answers as new
  ten-shot responses.
- Report macro Recall@10, paired 95% case-bootstrap intervals (20,000 draws,
  seed 20261030), valid/invalid counts, and per-case improved/worsened counts.
  The main follow-up contrast is periods versus no filler. Also report counting
  versus no filler, each condition's ten-minus-one-shot change, and the paired
  difference in filler gains between shot counts. All intervals are unadjusted
  exploratory intervals; no tuning based on results.

Example count and filler-copy count match arithmetic, but total input tokens,
task instructions, answer format and message roles still differ. The additional
examples are in-domain gold annotations, whereas the original example is
synthetic; changing shot count also changes example content and supervision.
The new examples ask for one/two annotated positives rather than a complete
top-ten ranking. Those limits prevent attributing every difference solely to
shot count. Saved controls were collected on earlier days, so provider-side
temporal variation and repeated-generation variability are not controlled.
