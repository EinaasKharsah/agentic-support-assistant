# Evaluation log

41 questions in `evaluation/eval_set.json`: 25 should be answered, 7 are vague
(should get a clarifying question), 9 are out of scope (should be declined).
Model: qwen2.5:3b, temperature 0. Each round changed ONE thing.

| Round | Change | Accuracy | Clarify correct | Unsafe answers |
|---|---|---|---|---|
| Baseline | Rules only: similarity scores decide | 83% | 2/7 | 0 |
| 1 | LLM chooses the action (answer / clarify / decline) | 80% | 2/7 | 0 |
| 2 | Out-of-scope card (Q10) hidden from the LLM | 78% | 1/7 | 0 |
| 3 | LLM describes (support question? which cards fit?), code routes | 88% | 5/7 | 0 |
| 4 | Second check ("A, B or either?") + retrieve 4 cards | 90% | 6/7 | 0 |
| 5 | Rewrite weak searches + clearer "support question" rule | 98% | 7/7 | 0 |
| 6 | Fix: rewrite step copied words from its prompt | **98%** | **7/7** | 0 |

From round 4, a second set of 20 hard cases (`evaluation/robustness_set.json`) was also run:

| Round | Robustness set | Errors |
|---|---|---|
| 4 | 70% | 6 over-declines (all 3 other-language questions, typos, one-word message, PayPal injection) |
| 5 | 85% | 2 over-clarify, 1 random text clarified instead of declined |
| 6 | **90%** | 1 over-clarify (typo), 1 over-decline (Arabic) |

Result files: `evaluation/results/1_baseline`, `2_hide_q10`, `3_model_describes_code_routes`,
`llm_qwen2.5-3b_all_check0.1.json` (round 4), and `*_check0.1_rewrite.json` (round 6).
The round 5 files were overwritten by round 6; its numbers come from the console output.

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

**Round 4: a second, focused question.** The model rarely noticed ambiguity by itself,
but it is good at choosing between two options. When another card scores within 0.1 of the
chosen one, the model now answers "A, B or either?". Retrieving 4 cards (instead of 3) keeps
3 candidates even after Q10 is removed. "I have a problem with my account" now clarifies.

**Robustness set: the embedding model is English-only.** All three questions in other
languages scored below 0.55 and were declined, and so were some typos.

**Round 5: rewrite weak searches.** When nothing scores 0.55, the model rewrites the message
as a short English search query and we search again. Only weak searches pay for this extra
call. German, Arabic, a typo and "Do you take Bitcoin?" were fixed. A prompt rule ("a support question is
true even if the cards do not mention the detail") fixed Trello being declined.

**Round 6: the accuracy number hid a problem.** Reading the saved rewrites showed the model
copying topic words from the rewrite prompt ("a help center about accounts, billing,
integrations and data") into its queries. "asdfgh qwerty zxcv" became "accounts billing
integrations data", which matches the cards, so random text was treated as a question.
Fix: no topic words in the prompt, and a `has_meaning` field first; the code skips the
rewrite when it is false. German now answers correctly too. Side effect: the 3b model
sometimes marks Arabic as meaningless, so Arabic is declined again (a safe error).
I stopped tuning here: there are no unsafe answers, and more changes based on 20 questions
would only fit this test set.

**Answer check (after round 6).** A test on the chat page showed an answer that left out
"Monthly plan charges are non-refundable after 48 hours" and added "Please contact support
to confirm your order ID" (the style of the prompt's own example). Code now checks every
answer: numbers, e-mails and menu paths in the answer must be in the card, and every number
in the card must be in the answer; otherwise the card text is shown. Main set with answers:
5 of 25 answers (20%) replaced, all for a left-out number, 0 invented facts. Decision
accuracy in that run: 100% (A07 was answered), so the main-set result varies between 98% and 100%.

## Held-out hard set (after all tuning)

20 harder questions in 8 new groups, written after all tuning, committed before the run
(commit b91b7dc), run once, and the agent was not changed afterwards. Result: **65%**
(13/20), Recall@1 78%. By group: needs reasoning 2/3, different wording 3/4, not covered 2/2,
near-miss 1/4, vague 2/3, two questions 1/1, French/Spanish 1/2, small talk 1/1.

- Main weakness: near-misses. "What is Netflix's refund policy?" and "How do I reset my Wi-Fi
  router password?" score 0.67 and 0.70, pass the guardrail, and the model calls them support
  questions. The Netflix reply showed our refund article without saying it is not Netflix's.
- "I got a strange email about my account" was answered with "Your account appears to be
  suspended": a guess where a clarifying question was needed.
- The Spanish question was declined: the rewrite returned Spanish, not English.
- Safe errors: two over-clarifications (negation; card failed and account locked).
- No reply invented a number, e-mail address or menu path.

This confirms that the development numbers (98 to 100%) were optimistic. They remain useful to
compare versions; 65% is the fairer estimate for new questions.

## Remaining errors (round 6)

- "My account got blocked after my card was declined": asks blocked account or cancel (too careful, but safe).
- "how do i chnage my emial adress": the second check wrote "clearly asking about changing email" but chose "either".
- "نسيت كلمة المرور" (Arabic, forgot password): declined, see round 6.

## Remaining errors (round 3, for history)

- German question: below the relevance threshold. The embedding model is English-only.
- "My account got blocked after my card was declined": asked a clarifying question (too careful).
- "Do you integrate with Trello?": reason says "about integrations", but marked as not a support question.
- "I want to stop paying": answered with "cancel subscription" instead of asking.
- "I have a problem with my account": the right cards (password, email) were not in the
  candidates. After removing Q10, only 2 of the top 3 cards were left. Next step: retrieve 4.

## Honest limits

- 41 questions is small. It is enough to compare versions, not to prove real-world quality.
- We looked at the errors on all 41 questions (dev and test) while improving the agent,
  so the numbers are optimistic. A new, unseen question set would give a fairer number.
- One person wrote the questions and the expected answers.
