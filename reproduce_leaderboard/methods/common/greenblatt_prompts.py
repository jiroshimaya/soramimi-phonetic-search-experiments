"""Prompt helpers from Ryan Greenblatt's no_cot_math_public, MIT licensed.

Source: commit 7f4d3e4e2f775db1ef45986c4996be9d7a34c7e1,
eval_math_no_reasoning.py. See data/greenblatt_gen_arithmetic/LICENSE.
Only self-contained prompt helpers are included; no client setup runs on import.
"""

import random

LOREM_IPSUM_WORDS = """lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum""".split()


def select_few_shot_problems(problems, k_shot=10):
    """
    Select k_shot problems for few-shot prompting:
    - First two problems
    - Middle problem
    - 5th from last
    - Last problem

    Returns: list of (index, problem) tuples in randomized order (but consistent across runs)
    """
    n = len(problems)

    assert n > k_shot, "Not enough problems to select few-shot examples."
    assert k_shot >= 2

    indices = [
        0,
        *[round(n * (i + 1) / (k_shot - 1)) for i in range(k_shot - 2)],
        n - 1,
    ]

    assert len(indices) == k_shot
    assert len(set(indices)) == k_shot, "Few-shot indices are not unique."

    # Get problems
    few_shot = [(i, problems[i]) for i in indices]

    # Randomize order with fixed seed for consistency across runs
    rng = random.Random(42)
    rng.shuffle(few_shot)

    return few_shot, set(indices)


def build_user_message(
    problem,
    repeat_problem: None | int = None,
    filler_tokens: None | int = None,
    filler_type: str = "numbers",
):
    instruction = "You will be given a math problem. Answer immediately using the format 'Answer: [ANSWER]' where [ANSWER] is just the numerical answer, nothing else. No explanation, no words, no reasoning, just the number."

    STATE_REPEAT_PROBLEMS = False
    EXPLAIN_REPEATS = False

    if STATE_REPEAT_PROBLEMS and repeat_problem is not None:
        instruction += (
            f" The text of the problem will be repeated {repeat_problem} times"
            + (
                " (so that you as an LLM have more space to think about the solution)."
                if EXPLAIN_REPEATS
                else "."
            )
        )

    if filler_tokens is not None:
        if filler_type == "numbers":
            instruction += f" After the problem, there will be filler tokens (counting from 1 to {filler_tokens}) to give you extra space to process the problem before answering."
        elif filler_type == "ellipsis":
            instruction += f" After the problem, there will be filler tokens ({filler_tokens} ellipses) to give you extra space to process the problem before answering."
        elif filler_type == "lorem":
            instruction += f" After the problem, there will be filler tokens ({filler_tokens} lorem ipsum words) to give you extra space to process the problem before answering."

    USE_REP_TEXT = True
    IDX_REP_TEXT = True

    def rep_text(idx):
        if not USE_REP_TEXT:
            return ""
        if repeat_problem is None or idx == 0:
            return ""
        else:
            if IDX_REP_TEXT:
                return f" (repeat #{idx + 1})"
            else:
                return " (repeat)"

    out = f"{instruction}\n\n" + (
        "\n\n".join(
            [
                f"Problem{rep_text(idx)}: {problem['problem']}"
                for idx in range(repeat_problem if repeat_problem is not None else 1)
            ]
        )
    )

    # Add filler tokens if specified
    if filler_tokens is not None:
        if filler_type == "numbers":
            filler = " ".join(str(i) for i in range(1, filler_tokens + 1))
        elif filler_type == "ellipsis":
            filler = " ".join(["..."] * filler_tokens)
        elif filler_type == "lorem":
            # Repeat lorem ipsum words as needed to get the right count
            words = []
            for i in range(filler_tokens):
                words.append(LOREM_IPSUM_WORDS[i % len(LOREM_IPSUM_WORDS)])
            filler = " ".join(words)
        else:
            raise ValueError(f"Unknown filler_type: {filler_type}")
        out += f"\n\nFiller: {filler}"

    # if repeat_problem is not None and repeat_problem > 5:
    #     out += f"\n\nNow answer immediately using the format 'Answer: [ANSWER]' where [ANSWER] is just the numerical answer, nothing else. No explanation, no words, no reasoning, just the number."

    # print("-----")
    # print(out)
    # print("-----")

    return out


def build_few_shot_messages(
    few_shot_problems,
    repeat_problem: None | int = None,
    filler_tokens: None | int = None,
    filler_type: str = "numbers",
    cache: bool = False,
    disable_prefill: bool = False,
):
    """Build the few-shot messages as user/assistant pairs.

    Args:
        for_openai_chat: If True, include "Answer:" at end of user message (for models without prefill support)
    """
    messages = []

    for idx, problem in few_shot_problems:
        user_text = build_user_message(
            problem,
            repeat_problem=repeat_problem,
            filler_tokens=filler_tokens,
            filler_type=filler_type,
        )
        if disable_prefill:
            user_text += "\n\nAnswer:"

        # User message with instruction and problem
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text,
                    }
                ],
            }
        )
        # Assistant message with just the answer
        messages.append(
            {
                "role": "assistant",
                "content": f"Answer: {problem['answer']}"
                if not disable_prefill
                else str(problem["answer"]),
            }
        )
    if cache:
        assert len(few_shot_problems) > 0
        messages[-2]["content"][0]["cache_control"] = {"type": "ephemeral"}

    return messages
