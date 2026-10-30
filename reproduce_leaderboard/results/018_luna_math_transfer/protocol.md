# Predeclared filler-token validation and transfer protocol

The first question is whether GPT-6 Luna benefits from the source study's filler
procedure on its public Gen-Arithmetic task. The second is whether the same
filler constructions improve phonetic reranking. Improvements are not assumed.

## Arithmetic validation

Source: [Greenblatt's study](https://www.lesswrong.com/posts/NYzYJ2WoB74E6uj9L)
and the MIT-licensed [public implementation](https://github.com/rgreenblatt/no_cot_math_public/tree/7f4d3e4e2f775db1ef45986c4996be9d7a34c7e1).
The public arithmetic dataset has 3,000 generated Python expressions. The
competition-math set behind the highlighted 45%→51% result is not public, so
this experiment cannot reproduce that particular score.

- Freeze 10 demonstrations using the source selector, including its seed 42.
- Exclude every demonstration from the evaluation set. Take the first 300 and
  last 300 remaining source problems: 600 fixed, distinct test cases.
- Validate all selected reference answers independently with an arithmetic AST
  evaluator supporting only the source's integer operators.
- Copy the source's instruction, OpenAI answer cue, and 10 alternating chat
  demonstrations. Demonstrations receive the same filler as the target.
- Conditions: none; counting 1–100; counting 1–300; 300 space-separated `...`
  strings (the source implementation's `ellipsis` condition).
- Primary comparison: counting 1–100 versus none. The source's arithmetic study
  uses 100 as its selected filler length. The other comparisons are exploratory.
- Model: API ID `gpt-6-luna`, `reasoning_effort=none`, `temperature=0`, maximum
  100 completion tokens, no structured-output schema. Record the returned model
  version and verify zero reported reasoning tokens.
- One first answer per case and condition: 2,400 requests. Save provider responses
  before parsing. Invalid, incomplete, or explanatory answers count as incorrect.
  A parsing failure does not trigger regeneration.
- Reuse cached answers only when their full request hashes match. Resume missing
  requests after transport/service failures; never repeat successful provider
  responses to select a better answer.
- Exact-match numeric accuracy, paired 20,000-draw case bootstrap intervals,
  and an exact two-sided McNemar test. Seed 20261030. No tuning on these answers.

This preserves the public task, filler material, and 10-shot procedure while
changing the model and excluding demonstration/test overlap. It is a cross-model
behavioral validation, not a reproduction of the original model's score or
white-box hidden-state analysis.

## Phonetic transfer

After the arithmetic result is available, compare the same four conditions with
the same model on all 150 queries of the pinned phonetic dataset. Use its existing
`detailed` instruction and single example, inserting filler before both the
example answer and target answer. Keep the established JSON ranking schema and
return ten distinct indices. Measure macro Recall@10. This evaluates reranking
with all positives included among 100 candidates, not speech recognition or
retrieval from the full lexicon. Differences in task, example count, and output
format must remain explicit in the article.

## Article sequence

1. Explain the filler-token question and distinguish filler-item counts from
   tokenizer counts.
2. Describe the source tasks, available data, and exact reproduction limits.
3. Report the arithmetic validation, including unchanged/worsened examples and
   uncertainty. If uplift is absent, state that this model did not validate the
   expected positive control.
4. Present phonetic transfer with the same model and filler definitions.
5. Separate behavioral measurements from claims about internal computation, and
   discuss quality, additional input tokens, and latency as practical tradeoffs.

The article's conclusion follows the measured outcomes; successful arithmetic
validation is not a predetermined conclusion.

## Follow-up model comparison

GPT-6.1 Terra is an additional model evaluated on the same 600 arithmetic cases,
10 demonstrations, and baseline/counting-100 pair, with all parameters preserved
except model ID. This follow-up is analyzed separately from the original Luna
protocol; its model was selected after observing Luna's results. Both model
results are retained. No conditions or cases are tuned on Terra's answers.
