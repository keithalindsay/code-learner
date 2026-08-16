"""Minimal-11b measurement: source-only vs assertions-on retrieval on kalshi-bot.

Scores SUBJECT-symbol retrieval (the gold labels symbols) in two configs on the same
queries: source-only (lexical+dense+graph) and assertions-on (adds served, judged tier-2
claims, whose value is that they PROMOTE their subject symbol into the results). Reports
the paired delta_ci from the tested ablation machinery, on the full 62-query gold set and
on the 38-query baseline-failure subset separately.

Assertions-on uses PRODUCTION_POLICY, so only claims with an independent supporting verdict
and live-verified citations are served -- the gate is part of the measurement.
"""
import json
from pathlib import Path

from codelearner import db
from codelearner.cli.main import _default_embedder
from codelearner.eval.ablation import _score
from codelearner.eval.goldset import GoldQuery
from codelearner.retrieve.mixed import search_candidates
from codelearner.retrieve.types import SourceCandidate

INDEX = "/tmp/claude-1000/-home-keith-AIWorkspace/67394a11-746a-4204-9603-0d9ca528ab0c/scratchpad/kalshi-11b-index.db"
ROOT = Path("/home/keith/projects/kalshi-bot")
GOLD = "/home/keith/projects/code-learner/codelearner/eval/gold/hand_kalshi_bot.json"
SUBSET = "/tmp/claude-1000/-home-keith-AIWorkspace/67394a11-746a-4204-9603-0d9ca528ab0c/scratchpad/baseline_subset.json"
K = 10

conn = db.connect(INDEX, check_schema=False)
indexed = {r["qualname"] for r in conn.execute("SELECT qualname FROM symbols")}
emb = _default_embedder("Qwen/Qwen3-Embedding-0.6B")

n_served = conn.execute(
    "SELECT count(*) FROM assertions a WHERE a.status='active' AND EXISTS "
    "(SELECT 1 FROM verdicts v WHERE v.assertion_id=a.id AND v.verdict='supported')"
).fetchone()[0]
n_total = conn.execute("SELECT count(*) FROM assertions").fetchone()[0]
print(f"assertions in index: {n_total} total, {n_served} active+supported (servable)\n")

gold = json.load(open(GOLD))
subset = json.load(open(SUBSET))
fail_queries = {r["query"] for r in (subset["miss"] + subset["buried"])}

def _sources(query, use_assertions):
    res = search_candidates(conn, ROOT, query, k=K, embedder=emb, use_assertions=use_assertions)
    return [c for c in res.candidates if isinstance(c, SourceCandidate)]

def _measure(queries, label):
    src_pairs, asrt_pairs = [], []
    for q in queries:
        present = tuple(r for r in q["relevant"] if r in indexed)
        if not present:
            continue
        gq = GoldQuery(query=q["query"], relevant=present, repo="kalshi-bot",
                       query_id=q["query"][:40])
        src_pairs.append((gq, _sources(q["query"], False)))
        asrt_pairs.append((gq, _sources(q["query"], True)))
    src = _score(f"source-only [{label}]", src_pairs)
    asrt = _score(f"assertions-on [{label}]", asrt_pairs)
    print(f"=== {label}: n={len(src_pairs)} ===")
    print(f"{'config':<22}{'hit@5':>8}{'recall@5':>10}{'mrr':>8}")
    for c in (src, asrt):
        print(f"{c.name.split('[')[0].strip():<22}{c.hit_at(5):>8.3f}{c.recall_at(5):>10.3f}{c.mrr:>8.3f}")
    for metric in ("hit", "mrr"):
        lo, hi = asrt.delta_ci(src, metric=metric, k=5)
        sign = "significant" if (lo > 0 or hi < 0) else "n.s. (CI spans 0)"
        print(f"  delta {metric}@5 (assertions - source): [{lo:+.3f}, {hi:+.3f}]  {sign}")
    print()
    return src, asrt

all_q = gold["queries"]
fail_q = [q for q in all_q if q["query"] in fail_queries]
_measure(all_q, "full gold set (62)")
_measure(fail_q, "baseline-failure subset (38)")
