"""Per-query rank diff: source-only vs assertions-on, to validate the mechanism."""
import json
from pathlib import Path
from codelearner import db
from codelearner.cli.main import _default_embedder
from codelearner.retrieve.mixed import search_candidates
from codelearner.retrieve.types import SourceCandidate, AssertionCandidate

INDEX = "/tmp/claude-1000/-home-keith-AIWorkspace/67394a11-746a-4204-9603-0d9ca528ab0c/scratchpad/kalshi-11b-index.db"
ROOT = Path("/home/keith/projects/kalshi-bot")
GOLD = "/home/keith/projects/code-learner/codelearner/eval/gold/hand_kalshi_bot.json"
K = 10
conn = db.connect(INDEX, check_schema=False)
indexed = {r["qualname"] for r in conn.execute("SELECT qualname FROM symbols")}
emb = _default_embedder("Qwen/Qwen3-Embedding-0.6B")
gold = json.load(open(GOLD))

def rank_and_claim(query, present, use_assertions):
    res = search_candidates(conn, ROOT, query, k=K, embedder=emb, use_assertions=use_assertions)
    src = [c.qualname for c in res.candidates if isinstance(c, SourceCandidate)]
    r = next((i for i, qn in enumerate(src, 1) if qn in present), None)
    n_claims = sum(1 for c in res.candidates if isinstance(c, AssertionCandidate))
    return r, n_claims

recovered, displaced, unchanged = [], [], []
for q in gold["queries"]:
    present = set(x for x in q["relevant"] if x in indexed)
    if not present:
        continue
    rs, _ = rank_and_claim(q["query"], present, False)
    ra, nc = rank_and_claim(q["query"], present, True)
    def top5(r): return r is not None and r <= 5
    if not top5(rs) and top5(ra):
        recovered.append((q["query"], rs, ra, nc))
    elif top5(rs) and not top5(ra):
        displaced.append((q["query"], rs, ra, nc))
    else:
        unchanged.append((q["query"], rs, ra))

print(f"RECOVERED (source miss/bury -> assertions top5): {len(recovered)}")
for query, rs, ra, nc in recovered:
    print(f"  src={rs} -> asrt={ra}  (claims served for query: {nc})  {query[:64]}")
print(f"\nDISPLACED (source top5 -> assertions worse): {len(displaced)}")
for query, rs, ra, nc in displaced:
    print(f"  src={rs} -> asrt={ra}  (claims served: {nc})  {query[:64]}")
print(f"\nUNCHANGED: {len(unchanged)}")
