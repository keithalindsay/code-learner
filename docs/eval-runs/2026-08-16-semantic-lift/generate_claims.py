"""Targeted claim generation for the minimal-11b measurement.

Generate tier-2 claims for EXACTLY the subjects of the baseline-failure subset (the
queries source-only search missed/buried), using the real llama3.1:8b generator via the
`learn` pipeline's `candidates=` override. Private symbols are included (many targets are
private helpers). Claims are drafted and admitted (they must cite real evidence spans);
judging happens in the next step.
"""
import json
import sqlite3
import sys
from pathlib import Path

from codelearner import db
from codelearner.generate import OllamaClaimGenerator, learn
from codelearner.generate.pipeline import Candidate

INDEX = "/tmp/claude-1000/-home-keith-AIWorkspace/67394a11-746a-4204-9603-0d9ca528ab0c/scratchpad/kalshi-11b-index.db"
ROOT = Path("/home/keith/projects/kalshi-bot")
SUBSET = "/tmp/claude-1000/-home-keith-AIWorkspace/67394a11-746a-4204-9603-0d9ca528ab0c/scratchpad/baseline_subset.json"

subset = json.load(open(SUBSET))
subjects = sorted({s for rec in (subset["miss"] + subset["buried"]) for s in rec["relevant_present"]})
print(f"failure subset: {len(subset['miss'])} miss + {len(subset['buried'])} buried; "
      f"{len(subjects)} distinct subject symbols to draft claims for", flush=True)

conn = db.connect(INDEX, check_schema=False)
placeholders = ",".join("?" * len(subjects))
rows = conn.execute(
    f"SELECT s.id, s.qualname, s.kind, s.name, f.path, s.line_start, s.line_end "
    f"FROM symbols s JOIN files f ON f.id = s.file_id WHERE s.qualname IN ({placeholders})",
    subjects,
).fetchall()
candidates = [
    Candidate(
        symbol_id=int(r["id"]), qualname=str(r["qualname"]), kind=str(r["kind"]),
        path=str(r["path"]), line_start=int(r["line_start"]), line_end=int(r["line_end"]),
    )
    for r in rows
]
found = {c.qualname for c in candidates}
missing = [s for s in subjects if s not in found]
print(f"resolved {len(candidates)} candidates in index; {len(missing)} subjects not found "
      f"as symbols (e.g. nested): {missing[:5]}", flush=True)

gen = OllamaClaimGenerator(model="llama3.1:8b", host="http://localhost:11434")


def _progress(p):
    # one line per symbol as it completes
    r = getattr(p, "result", None)
    outcome = getattr(r, "outcome", "?") if r else getattr(p, "phase", "?")
    print(f"  [{getattr(p,'position','?')}/{getattr(p,'total','?')}] "
          f"{getattr(p,'candidate',None) and p.candidate.qualname}  -> {outcome}", flush=True)


report = learn(conn, ROOT, gen, candidates=candidates, on_progress=_progress)
print("\n=== LEARN REPORT ===", flush=True)
for attr in ("considered", "admitted", "skipped_existing", "refused", "errored"):
    print(f"  {attr}: {getattr(report, attr, 'n/a')}", flush=True)
n_assert = conn.execute("SELECT count(*) FROM assertions").fetchone()[0]
print(f"  assertions now in index: {n_assert}", flush=True)
