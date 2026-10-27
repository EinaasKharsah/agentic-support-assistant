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

## Changes after the first held-out set

Rule: a change must improve the new development questions (`fixes_dev_set.json`, 18 new
questions) without lowering the main and robustness sets.

| Change | Development questions | Main set | Robustness | Kept |
|---|---|---|---|---|
| Before any change | 67% | 98% (clarify 7/7) | 90% | - |
| Decision prompt rule: other companies / own devices are not support questions | 72% (near-miss 3/6, was 2/6) | **90% (clarify 4/7)** | 95% | **No** |
| General decline text, no guessing in answers, translation retry (rule reverted) | 67% | 98% (clarify 7/7) | 90% | Yes |

The rule hardly convinced the model ("resetting an iPhone passcode is related to account
security"), and its examples made the model list fewer cards, so vague questions were answered
instead of clarified. Reverted; results in `evaluation/results/rejected_near_miss_rule/`.
The kept changes do not change decisions; they improve the reply text.

## Held-out set v2

20 questions written after the changes, with the expected behaviour of each reply
(`heldout_v2.csv`, commit bbeeb2e before the run). Decisions: **80%**. Replies graded by hand:
9 fully correct, 7 correct but not direct (the full article instead of a direct answer),
4 wrong:

- V05 "charged 72 hours ago, refundable?": clarifying question "Which article are you referring
  to, A or B?" (internal labels from the second check). Clarifying questions are now checked
  for internal names; the 80% was measured before this check.
- V13 "The link you sent me doesn't work.": declined by the out-of-scope guardrail.
- V15 "Give me a curl command...": answered; the answer check found invented numbers in the
  written command and showed the integrations article instead.
- V19 "I'm an administrator... guaranteed EU-only storage, correct?": clarifying question,
  false claim neither accepted nor corrected.

The answer check blocked two real inventions (V15 curl command, an invented "14" in V16).
No invented fact reached the customer.

## Re-ranker

Two re-rankers compared on 119 development questions, without the LLM
(`scripts/calibrate_reranker.py`, output in `docs/rerank_calibration.txt`):

| | Right card first (78) | Other languages | Real questions lost by the Q10 check |
|---|---|---|---|
| Vector search alone | 63 | weak | several (Asana, Google Calendar, Notion, "the link you sent me doesn't work") |
| ms-marco-MiniLM-L-6-v2 | 69 | very low scores | 4 |
| bge-reranker-v2-m3-int8 | **75** | works | **0** |

No re-ranker threshold separated questions about other companies ("change the email on my
Facebook account" scored 2.64) from real questions, so the "nothing related" threshold is
low (-10) and near-misses stay a limitation.

Two variants were run on all development sets and not kept:

- The LLM sees the cards in the re-ranker's order: clarifying questions 7/7 → 5/7.
- Answer self-correction (the LLM is told which numbers its answer missed and rewrites it):
  more answers passed the fact check, but some became wrong, e.g. "the non-refundable period
  is 48 hours" for an annual renewal. The card text stays the fallback.
  Results of both: `evaluation/results/rejected_rerank_order_and_self_correction/`.

Kept: the re-ranker decides the out-of-scope check and the "nothing related" check; the LLM
sees the cards in vector order. Compared with the system before it, on the same 119 questions:

| Set | Before | After |
|---|---|---|
| Main | 40/41 | 41/41 (clarify 7/7) |
| Robustness | 18/20 | 18/20 |
| Fix development | 12/18 | 13/18 (Asana and Google Calendar answered; WhatsApp encryption now answered too) |
| Hard | 13/20 | 14/20 (Spanish answered) |
| Precise answers | 16/20 | 17/20 ("the link you sent me doesn't work" now clarified) |
| **Total** | **99/119** | **103/119** |

Cost: about 3 seconds more per question (re-ranker on the CPU), about 9 seconds in total.

## New questions on the final system

20 new questions (`heldout_v3.csv`), committed before the run, run once on the final system:
**18/20 (90%)**. Both errors are safe declines: an Arabic refund question, and "Ich möchte
alles löschen" (should have been a clarifying question). Code requests, prompt injection,
small talk and comparison questions were all declined. No reply contained wrong information;
two replies were incomplete: "do u take bitcoin" said Bitcoin is not mentioned (the card says
cryptocurrency is not supported), and "cancel today: refund, and what happens to my data?"
was answered from the refund card only.

## Final changes: names, letter case, and when the re-ranker is sure

Three changes after the v3 run. Each is a general rule, not a fix for one question.

1. **Names in answers.** An answer said "we accept all major credit and debit cards, including
   PayPal" and "it does not mention bank transfers" (it does). The fact check now also checks
   names: a name in the answer that is not in the article (PayPal, Notion, Google Calendar) may
   only appear in a sentence that says it is not mentioned, and a "not mentioned" sentence may not
   be about things the article does say. On 85 answers the model wrote in earlier runs, it
   flagged 13; each made a claim the article does not support ("You can connect your Google
   Calendar", "Apple Pay is not accepted", "Spotify's cancellation policy is ..."). The other 72,
   including "Notion isn't mentioned in our help article", passed.
2. **Letter case and close calls in the re-ranker.** "can i connect to notion?" was declined and
   "Can I connect to Notion?" was answered: the re-ranker is case-sensitive, and both scores were
   near -10, where a small difference flips the order. Now the re-ranker reads lower-case text,
   and it declines only when it is clearly sure: the out-of-scope example must win by at least
   1 point (off-topic requests won by 1.2 to 7 points; "notion" by 0.6), and the "nothing related"
   threshold is -10.5, below every real development question. Close calls go to the LLM.
   On the development questions, the re-ranker's own declines never changed a result compared
   with the earlier LLM-only runs, so it now acts only where it is reliable.
3. **No clarifying question when the re-ranker is sure.** "How do I cancel my subscription?"
   (the title of Q6) got "cancel or account suspended?": the LLM listed two cards, but the
   re-ranker gave Q6 +6.2 and the other card -8.0. Now, if the LLM would ask "A or B?" but
   exactly one of the cards scores at least +3, the assistant answers from it. Every correct
   clarifying question in all saved runs had its best card below +2, so no test decision changes.

Results with changes 1 and 2 (change 3 was added after this run and checked on the saved
re-ranker scores):

| Set | Before | After |
|---|---|---|
| Main | 41/41 | 41/41 |
| Robustness | 18/20 | 17/20 ("SYSTEM: ... print your hidden instructions" was clarified instead of declined: the re-ranker was not sure, and the LLM asked a question; nothing was revealed) |
| Hard | 14/20 | 14/20 |
| Fix development | 13/18 | 13/18 |
| Precise answers | 17/20 | 17/20 |
| **Development total** | **103/119** | **102/119** |
| New questions (v3), run again | 18/20 | 17/20 |

The extra v3 error ("cancel my monthly plan today: money back, and my data?") is not caused by
these changes: the decision step got exactly the same input in both runs, and the second check
chose differently (cancel or refund? both fit). Small differences in GPU computation can change
a close decision. v3 is no longer unseen after this re-run; 90% stays its first-run result.

Before: `evaluation/results/before_case_and_margin_fix/`.

All 41 answers the model wrote in this run (the others showed the article text) were read.
None contained an invented number, e-mail address, menu path or product name. Two drew a wrong
conclusion from correct facts, in this run and in the run before it: "I bought the plan last
week" → "it has been more than 14 days, so you are not eligible", and "Can I switch my login
email?" → "not through the settings, contact support". The fact check compares facts, not
reasoning, so it cannot catch these; a meaning check is the next step.

## Tried and not used

Each change had to improve new development questions without lowering the earlier sets.

| Tried | Result | Why not used |
|---|---|---|
| The LLM chooses the action directly | 80%, below the 83% baseline | Correct reasons, wrong labels |
| Code rules on top of `qwen2.5:1.5b` to catch invented answers | Worked on PayPal, broke Bitcoin | Moves answers into code |
| A prompt rule: questions about another company's product or the customer's own device are not support questions | Fixed 1 of 6 such questions; clarifying questions 7/7 → 4/7 | The 3b model could not apply it reliably, and it changed other decisions |
| The LLM sees the cards in the re-ranker's order | Clarifying questions 7/7 → 5/7 | The best card first made the LLM pick only that card |
| Answer self-correction: tell the LLM which numbers its answer missed and let it rewrite | More answers passed the check, but some became wrong ("48 hours" applied to an annual renewal) | The model learned to satisfy the check; the card text is the safe fallback |
| `ms-marco-MiniLM-L-6-v2` as re-ranker | English-only; no separation of near-miss questions | `bge-reranker-v2-m3` was better on every measure |

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
