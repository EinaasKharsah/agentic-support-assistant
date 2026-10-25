# Model comparison (answer generation)

Same prompt, same retrieved card, temperature 0. Tested on a laptop with an
NVIDIA GTX 1650 (4 GB VRAM). The key test: questions about things the
knowledge base does **not** mention. A good answer says "not mentioned" and
shares what the article does say. A bad answer invents a yes/no.

| Question | qwen2.5:0.5b | qwen2.5:1.5b | qwen2.5:3b |
|---|---|---|---|
| Can I pay with PayPal? | "Yes, PayPal is accepted" (hallucination) | "Yes, you can pay with PayPal" (hallucination) | "PayPal is not mentioned in our help article" + accepted methods |
| Do you integrate with Trello? | - | Copied the card, did not say Trello is not mentioned | "Trello isn't mentioned" + list of integrations |
| Do you take Bitcoin? | - | "No, Bitcoin is not supported" (correct, but short) | "Bitcoin is not supported" + accepted methods |
| Is my data stored in Germany? | - | - | Good EU residency info, but added "in multiple locations" (not in the card) |
| I forgot my password | - | Correct | Correct |

Notes:
- With the shorter first prompt, 1.5b answered "No, PayPal is not accepted"
  (also not in the card). Adding rules and an example to the prompt made 1.5b
  *worse*: small models are very sensitive to prompt changes.
- I also tried adding code rules on top of 1.5b (detect unknown names, use
  fixed answers). It worked but moved the answers into code, was brittle
  (it broke the correct Bitcoin answer) and is not what an agent should do.
- **Decision:** use qwen2.5:3b, the smallest model that passed without extra
  rules. It fits fully in 4 GB of GPU memory (~2 GB). Answers took about 4 s.
- Remaining risk: 3b can still add small details (see the Germany example).
  This is measured in the evaluation set.
