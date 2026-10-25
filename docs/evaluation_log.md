# Evaluation log

41 questions in `evaluation/eval_set.json`: 25 should be answered, 7 are vague
(should get a clarifying question), 9 are out of scope (should be declined).
Model: qwen2.5:3b, temperature 0. Each round changed ONE thing.

| Round | Change | Accuracy | Clarify correct | Unsafe answers |
|---|---|---|---|---|
| Baseline | Rules only: similarity scores decide | 83% | 2/7 | 0 |
| 1 | LLM chooses the action (answer / clarify / decline) | 80% | 2/7 | 0 |
| 2 | Out-of-scope card (Q10) hidden from the LLM | 78% | 1/7 | 0 |
| 3 | LLM describes (support question? which cards fit?), code routes | **88%** | **5/7** | 0 |

Result files: `evaluation/results/1_baseline`, `2_hide_q10`, `3_model_describes_code_routes`.

## What each round taught us

**Scores alone cannot detect vague questions.** A clear question ("Can I get my
money back?") had a gap of 0.049 between the top two cards; vague ones ("I want
to stop paying") had gaps of 0.053 to 0.064. No threshold separates them.

**Round 1: the model's reason and action did not match.** For "Do you take
Bitcoin?" it wrote that Q4 (payment methods) fits, then chose "decline". Its
reason mentioned Q10: "Q10 is marked OUT OF SCOPE and should be declined". Q10 is
very generic, so it appears in the top 3 for most questions.

**Round 2: hiding Q10 fixed Bitcoin and CSV export,** but the model still chose
"decline" for "Which credit cards can I use?" while writing "Q4 accepts all major
credit cards". The 3B model does not reliably turn its own reasoning into an
action label. It also copied a prompt phrase ("This is the most common case") as
its argument to answer instead of asking.

**Round 3: ask simpler questions.** The model now fills in: is this a support
question, and which cards could the customer mean (all of them)? Code maps this:
not support -> decline, one card -> answer, 2-3 cards -> clarify. The model still
does all the understanding; the code only routes. Listing *all* possible cards
made the model consider other meanings, and clarify went from 1/7 to 5/7.

## Remaining errors (round 3)

- German question: below the relevance threshold. The embedding model is English-only.
- "My account got blocked after my card was declined": asked a clarifying question (too careful).
- "Do you integrate with Trello?": reason says "about integrations", but marked as not a support question.
- "I want to stop paying": answered with "cancel subscription" instead of asking.
- "I have a problem with my account": the right cards (password, email) were not in the
  candidates. After removing Q10, only 2 of the top 3 cards were left. Next step: retrieve 4.

## Honest limits

- 41 questions is small. It is enough to compare versions, not to prove real-world quality.
- We looked at the errors on all 41 questions (dev and test) while improving the agent,
  so 88% is optimistic. A new, unseen question set would give a fairer number.
- One person wrote the questions and the expected answers.
