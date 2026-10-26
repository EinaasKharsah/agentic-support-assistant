"""Compare re-ranking models and choose the threshold, using the development questions.

No LLM is needed, so this is fast. For every question it runs the vector search, lets the
re-ranker score the 4 articles found, and checks two things:
  1. Ranking: is the right article first after re-ranking (compared with vector search alone)?
  2. Declining: for each threshold, how many questions would be declined correctly
     ("no article really answers this") and how many real questions would be lost?

Usage:
  python scripts/calibrate_reranker.py
  python scripts/calibrate_reranker.py Xenova/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-v2-m3-int8
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.embeddings import FastEmbedEmbedder  # noqa: E402
from app.knowledge_base import load_knowledge_base  # noqa: E402
from app.reranker import CrossEncoderReranker  # noqa: E402
from app.vector_store import VectorStore, make_client  # noqa: E402

DEFAULT_MODELS = ["Xenova/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-v2-m3-int8"]
SETS = ["eval_set.json", "robustness_set.json", "fixes_dev_set.json", "hard_set.json"]


def load_items() -> list[dict]:
    items = []
    for name in SETS:
        for item in json.loads((ROOT / "evaluation" / name).read_text(encoding="utf-8")):
            items.append({**item, "set": name.split("_")[0]})
    v2 = ROOT / "evaluation" / "heldout_v2.csv"
    if v2.exists():
        with v2.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file, delimiter=";"):
                actions = row["expected"].split("|")
                items.append({"id": "V", "set": "heldout_v2", "question": row["question"],
                              "expected_action": actions if len(actions) > 1 else actions[0],
                              "expected_cards": row["cards"].split()})
    return items


def accepted(item: dict) -> list[str]:
    expected = item["expected_action"]
    return expected if isinstance(expected, list) else [expected]


def main() -> None:
    # Windows consoles default to cp1252, which cannot print Arabic or Spanish questions.
    sys.stdout.reconfigure(encoding="utf-8")
    models = sys.argv[1:] or DEFAULT_MODELS
    entries = load_knowledge_base(ROOT / settings.knowledge_base_path)
    store = VectorStore(make_client(None), FastEmbedEmbedder(settings.embedding_model), "calibrate")
    store.index(entries)
    items = load_items()
    print(f"{len(items)} questions from {', '.join(SETS)} (+ heldout_v2 if present)\n")

    searches = [(item, store.search(item["question"], top_k=4)) for item in items]

    for model_name in models:
        reranker = CrossEncoderReranker(model_name)
        rows = []
        for item, matches in searches:
            scores = reranker.score(item["question"], [m.entry.to_text() for m in matches])
            ranked = sorted(zip(matches, scores), key=lambda pair: pair[1], reverse=True)
            in_scope = [(m, s) for m, s in ranked if not m.entry.out_of_scope]
            rows.append({
                "item": item,
                "vector_top": matches[0].entry.id,
                "rerank_top": ranked[0][0].entry.id,
                "best_in_scope": in_scope[0][0].entry.id if in_scope else None,
                "best_in_scope_score": in_scope[0][1] if in_scope else float("-inf"),
                "q10_top": ranked[0][0].entry.out_of_scope,
            })

        # 1. Ranking: questions that should be answered, is the right card first?
        answerable = [r for r in rows if "answer" in accepted(r["item"]) and r["item"]["expected_cards"]]
        vec_ok = sum(r["vector_top"] in r["item"]["expected_cards"] for r in answerable)
        rr_ok = sum(r["best_in_scope"] in r["item"]["expected_cards"] for r in answerable)
        print(f"=== {model_name}")
        print(f"Right article first (answerable questions, n={len(answerable)}): "
              f"vector search {vec_ok}, re-ranker {rr_ok}")

        # 2. Declining: should-decline vs must-not-decline, for several thresholds.
        must_decline = [r for r in rows if accepted(r["item"]) == ["decline"]]
        must_keep = [r for r in rows if "decline" not in accepted(r["item"])]
        scores = sorted({round(r["best_in_scope_score"], 1) for r in rows})
        print(f"Should decline: {len(must_decline)}   Must NOT decline: {len(must_keep)}")
        print("threshold | declined correctly | real questions lost (Q10 first / below threshold)")
        for t in scores[:: max(1, len(scores) // 15)]:
            caught = sum(r["q10_top"] or r["best_in_scope_score"] < t for r in must_decline)
            lost_q10 = sum(r["q10_top"] for r in must_keep)
            lost_low = sum((not r["q10_top"]) and r["best_in_scope_score"] < t for r in must_keep)
            print(f"{t:9.1f} | {caught:2d}/{len(must_decline)} | {lost_q10 + lost_low:2d} ({lost_q10} / {lost_low})")

        print("\nScores per question (best in-scope article, Q10 first?):")
        for r in sorted(rows, key=lambda r: r["best_in_scope_score"]):
            exp = "/".join(accepted(r["item"]))
            print(f"  {r['best_in_scope_score']:7.2f} {'Q10' if r['q10_top'] else '   '} "
                  f"{exp:<15} {r['item']['set'][:10]:<10} {r['item']['question'][:70]}")
        print()


if __name__ == "__main__":
    main()
