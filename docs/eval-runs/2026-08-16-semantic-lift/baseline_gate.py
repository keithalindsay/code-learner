"""Baseline gate for minimal-11b: does source-only search FAIL on kalshi-bot hand-gold?

For each hand-authored gold query, run source-only retrieval (lexical+dense+graph, no
assertions) and classify:
  hit    - a relevant subject in top-5
  buried - a relevant subject present but rank 6..10
  miss   - no relevant subject in top-10
  stale  - none of the query's relevant qualnames exist in THIS index (excluded; a
           retriever cannot rank a symbol that is not indexed, so it is not a real miss)

The baseline-failure subset (miss + buried, restricted to relevant symbols that ARE in
the index) is what the semantic layer will get a chance to recover.
"""
import json
import sys
from pathlib import Path

from codelearner import db
from codelearner.cli.main import _default_embedder
from codelearner.retrieve.mixed import search_candidates
from codelearner.retrieve.types import SourceCandidate

INDEX = "/tmp/claude-1000/-home-keith-AIWorkspace/67394a11-746a-4204-9603-0d9ca528ab0c/scratchpad/kalshi-11b-index.db"
ROOT = Path("/home/keith/projects/kalshi-bot")
GOLD = "/home/keith/projects/code-learner/codelearner/eval/gold/hand_kalshi_bot.json"
OUT = "/tmp/claude-1000/-home-keith-AIWorkspace/67394a11-746a-4204-9603-0d9ca528ab0c/scratchpad/baseline_subset.json"
K = 10
BURY_RANK = 5

conn = db.connect(INDEX, check_schema=False)
indexed = {r["qualname"] for r in conn.execute("SELECT qualname FROM symbols")}
print(f"index has {len(indexed)} symbols", file=sys.stderr)

emb = _default_embedder("Qwen/Qwen3-Embedding-0.6B")

gold = json.load(open(GOLD))
queries = gold["queries"]

hit, buried, miss, stale = [], [], [], []
for q in queries:
    present = [r for r in q["relevant"] if r in indexed]
    if not present:
        stale.append(q["query"])
        continue
    res = search_candidates(conn, ROOT, q["query"], k=K, embedder=emb, use_assertions=False)
    ranked = [c.qualname for c in res.candidates if isinstance(c, SourceCandidate)]
    rank = next((i for i, qn in enumerate(ranked, 1) if qn in set(present)), None)
    rec = {
        "query": q["query"],
        "relevant_present": present,
        "coverage": q.get("coverage", []),
        "first_relevant_rank": rank,
        "top5": ranked[:5],
    }
    if rank is None:
        miss.append(rec)
    elif rank > BURY_RANK:
        buried.append(rec)
    else:
        hit.append(rec)

n_scored = len(hit) + len(buried) + len(miss)
print(f"\n=== BASELINE GATE (source-only, k={K}) ===")
print(f"total gold queries:   {len(queries)}")
print(f"stale (excluded):     {len(stale)}  (relevant symbols not in this index)")
print(f"scored:               {n_scored}")
print(f"  hit (rank<=5):      {len(hit)}")
print(f"  buried (rank 6-10): {len(buried)}")
print(f"  miss (absent<=10):  {len(miss)}")
gap = len(buried) + len(miss)
print(f"\nBASELINE-FAILURE SUBSET (miss + buried): {gap} of {n_scored} scored "
      f"({100*gap/n_scored:.0f}%)")

subset = miss + buried
json.dump({"miss": miss, "buried": buried, "n_scored": n_scored, "stale": len(stale)},
          open(OUT, "w"), indent=2)
print(f"\nwrote baseline-failure subset ({len(subset)} queries) -> {OUT}")
print("\n--- the failure subset (query -> present relevant subjects, coverage) ---")
for r in subset:
    tag = "MISS" if r["first_relevant_rank"] is None else f"rank {r['first_relevant_rank']}"
    print(f"[{tag}] {r['query'][:70]}")
    print(f"        subjects: {r['relevant_present']}  cov={r['coverage']}")
