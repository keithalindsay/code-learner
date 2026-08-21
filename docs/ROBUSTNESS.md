# Robustness: how it fails safe

[← code-learner](../README.md) · the case study

The design assumes the world misbehaves — a hostile repo, a rebuilt index under a live server, a file that moved after it was cited. Three mechanisms carry that, and each has its evidence elsewhere in the case study.

---

## Repository isolation

One SQLite file per repo (`.codelearner/index.db`). Cross-contamination is
**structurally impossible** — there is no shared store — and additionally enforced:
an index is pinned to one repo root and refuses a second. Add `.codelearner/` to the
indexed repo's `.gitignore`; nothing in it belongs in version control.

Source files come from `git ls-files` where available. That is the correctness path,
not a convenience: the first spike indexed `swarm-sync/.claude/worktrees/`, five
near-complete copies of the repo, and produced cross-copy edges binding a call in
one copy to a definition in another. A repo's own `.gitignore` already knows what is
real source.

Indexing a repo used to execute arbitrary commands out of that repo's own
`.git/config`. Git honours `core.fsmonitor` by *executing* it and reads it from the
config of the repo it is pointed at, so indexing a directory a second party could
write to ran their command, silently, while reporting a successful index. The
overrides now go on the command line, which is the only tier that outranks repo
config — a config-file mitigation loses to the file being defended against — plus
`GIT_CONFIG_NOSYSTEM`, which has no `-c` equivalent. `GIT_CONFIG_GLOBAL` is
deliberately left alone: `~/.gitconfig` is the one tier a hostile repo cannot write,
so blanking it pays a real cost against no attacker. Scoped honestly: `git clone` does
not transfer remote config, so this was never "clone a repo and get owned". It fired on
repos delivered as archives, vendored submodules, and agent-written directories.

## The adversarial gate

Tier-2 claims are admitted only through a gate that cannot be argued with — subject-exists, span-exists, bytes-still-hash — enforced identically at both the MCP door and the store door. It is measured by generating an adversarial corpus from what an index actually holds and mutation-testing every rule. The design of the store is in [Architecture](ARCHITECTURE.md#the-t2-assertion-store); the numbers, the two-door census, and the one paragraph on what the corpus *cannot* show are in [Results](RESULTS.md#the-gate).

## Staleness: verification at serve time, not on a timer

A served claim re-reads and re-hashes its cited bytes on every call; there is no background sweep with a window in which the index answers from code that no longer exists. Stale claims are marked, never deleted, and the rejected-and-stale sets are the only evidence the gate does anything. The mechanism, its two-stage fast path, and the failure modes it keeps apart are in [Architecture](ARCHITECTURE.md#the-staleness-engine).
