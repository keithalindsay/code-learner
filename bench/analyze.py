"""Variance, sizing, and paired intervals over the runs `harness.py` left on disk.

Reads `results/*.json` and never re-runs an agent. The three things it does, in the
order the honest version of this benchmark has to do them:

1. **Measure the variance first.** `variance_report` takes repeated runs of ONE task in
   ONE arm and reports the spread of tokens, cost, tool calls and wall-clock. Nothing
   downstream is meaningful without it: a design is sized against the noise it actually
   has, and agent-task noise is large enough that a four-run-per-arm median -- which is
   what the comparison benchmark publishes -- cannot see it at all. Four runs give
   three degrees of freedom; the sd estimate from three df has a 95% interval running
   roughly 0.6x to 2.9x the true value, so "four runs, median only" is not a small
   sample of the effect, it is no measurement of the noise whatsoever.

2. **Size the run from that variance**, using `codelearner.eval.ablation` rather than a
   second implementation of the same arithmetic: `paired_sd`-equivalent spread feeds
   `required_n`, `ci_half_width` and `design_effect`, and `CALIBRATION_FLOOR` says when
   the interval means what it says. Importing them is deliberate -- those functions
   carry measured calibration notes that a fresh copy of the formula would not.

3. **Bootstrap paired, clustered on task.** Runs of the same task share that task's
   difficulty, its repo, its file layout and its one prompt. Resampling runs
   independently would count 12 runs of one task as 12 independent facts and return an
   interval too narrow by roughly `sqrt(1 + (m-1) * ICC)`. `delta_ci` resamples TASKS.

## What this design can and cannot resolve

The metrics here are ratios of positive quantities with heavy right tails, not the
bounded [0,1] scores `ablation` was written against, so the effect size is expressed as
a RATIO and the sizing is done on `log(metric)` where a ratio becomes a difference and
the sd becomes roughly a coefficient of variation. `sizing_report` prints the resolvable
ratio for the n a budget buys, and prints it as a ratio ("this design cannot resolve
less than a 1.4x difference") rather than as a p-value, because the ratio is the thing a
reader can check against a published claim.

The cluster bootstrap has as many effective observations as there are TASKS. With fewer
than about five tasks the pooled interval is unstable however many runs sit inside them,
and `diagnostics` says so on the face of the report instead of leaving it to be noticed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from harness import RESULTS_DIR

from codelearner.eval.ablation import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CALIBRATION_FLOOR,
    ci_half_width,
    design_effect,
    required_n,
)

#: The cost metrics, in the order the report prints them. Wall-clock is last and
#: flagged: it is the noisiest of the four and codegraph's own README concedes a floor
#: effect on small repos, so a wall-clock win on a 75-file repo is measuring process
#: startup as much as it is measuring retrieval.
METRICS = ("total_tokens", "total_cost_usd", "n_tool_calls", "wall_s")
NOISY = frozenset({"wall_s"})

#: Tool calls spent locating deferred MCP tools are startup-time overhead, not index
#: work (see `harness` module docstring). Reported both ways so the choice is visible.
ADJUSTED = {"n_tool_calls_adj": ("n_tool_calls", "toolsearch_calls")}


def load(results_dir: Path = RESULTS_DIR, *, include_void: bool = False) -> list[dict]:
    """Every run on disk, `ok` ones only unless asked otherwise.

    A void run -- one whose MCP server never started -- is excluded by default and
    counted in the report. Silently averaging it in is the failure this whole harness
    is built to prevent: it looks like a cheap success.
    """
    root = Path(results_dir)
    runs = []
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".stream.jsonl"):
            continue
        # Directories under `results/` whose name starts with `_` hold exploratory and
        # verification runs -- pilots, config probes, harness self-checks. They are kept
        # as evidence but must never be swept into a headline average, so the default
        # sweep skips them and `--results-dir results/_exploratory/pilot` opts back in.
        if any(part.startswith("_") for part in path.relative_to(root).parts[:-1]):
            continue
        try:
            r = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict) or "arm" not in r or "task_id" not in r:
            continue
        if r.get("task_id") == "_verify":
            continue
        if include_void or r.get("ok"):
            runs.append(r)
    return runs


def _value(run: dict, metric: str) -> float:
    if metric in ADJUSTED:
        base, sub = ADJUSTED[metric]
        return float(run.get(base, 0)) - float(run.get(sub, 0))
    return float(run.get(metric, 0))


# ---------------------------------------------------------------------------------
# 1. Variance
# ---------------------------------------------------------------------------------


@dataclass
class Spread:
    """The distribution of one metric over repeated runs of one task in one arm."""

    metric: str
    n: int
    mean: float
    sd: float
    #: sd / mean. The scale-free number, and the one that feeds sizing: for a ratio
    #: comparison it is approximately the sd of `log(metric)`.
    cv: float
    minimum: float
    median: float
    maximum: float
    #: max / min. Blunt, and the one a reader checks a published median against.
    fold_range: float
    #: 95% interval for the sd itself, from the chi-square distribution. Printed
    #: because an sd from a handful of runs is an estimate with its own wide interval,
    #: and quoting it bare is how a four-run benchmark convinces itself it has measured
    #: something.
    sd_ci: tuple[float, float]


def _sd_ci(sd: float, n: int) -> tuple[float, float]:
    """95% CI for a normal sd from `n` observations, via a Wilson-Hilferty chi-square.

    Avoids a scipy dependency for one number. The cube-root approximation to the
    chi-square quantile is accurate to well under a percent for df >= 2, which is far
    finer than the point being made -- that with n=4 the interval spans a factor of
    nearly five.
    """
    df = n - 1
    if df < 1 or sd <= 0:
        return (0.0, float("inf"))

    def chi2_q(p: float) -> float:
        z = statistics.NormalDist().inv_cdf(p)
        return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3

    hi_chi, lo_chi = chi2_q(0.975), chi2_q(0.025)
    return (
        sd * math.sqrt(df / hi_chi) if hi_chi > 0 else 0.0,
        sd * math.sqrt(df / lo_chi) if lo_chi > 0 else float("inf"),
    )


def spread(values: Sequence[float], metric: str) -> Spread:
    vals = [float(v) for v in values]
    n = len(vals)
    mean = statistics.fmean(vals) if vals else 0.0
    sd = statistics.stdev(vals) if n > 1 else 0.0
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 0.0)
    return Spread(
        metric=metric,
        n=n,
        mean=mean,
        sd=sd,
        cv=(sd / mean) if mean else 0.0,
        minimum=lo,
        median=statistics.median(vals) if vals else 0.0,
        maximum=hi,
        fold_range=(hi / lo) if lo else float("inf"),
        sd_ci=_sd_ci(sd, n),
    )


def variance_report(runs: list[dict], task_id: str | None = None,
                    arm: str | None = None) -> str:
    """The spread of every metric over repeats, per (task, arm) cell.

    Run this BEFORE choosing n. It is the input to every sizing answer and the only
    part of the sizing that belongs to this benchmark rather than to arithmetic.
    """
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        if task_id and r["task_id"] != task_id:
            continue
        if arm and r["arm"] != arm:
            continue
        cells[(r["task_id"], r["arm"])].append(r)

    lines = ["VARIANCE -- repeated runs of one task in one arm", ""]
    for (tid, a), rs in sorted(cells.items()):
        if len(rs) < 2:
            continue
        lines.append(f"{tid}  arm={a}  n={len(rs)}")
        lines.append(
            f"  {'metric':<18} {'mean':>12} {'sd':>12} {'cv':>7} "
            f"{'min':>12} {'median':>12} {'max':>12} {'max/min':>8}  sd 95% CI"
        )
        for metric in (*METRICS, *ADJUSTED):
            s = spread([_value(r, metric) for r in rs], metric)
            flag = " *noisy" if metric in NOISY else ""
            lines.append(
                f"  {metric:<18} {s.mean:>12,.4g} {s.sd:>12,.4g} {s.cv:>7.1%} "
                f"{s.minimum:>12,.4g} {s.median:>12,.4g} {s.maximum:>12,.4g} "
                f"{s.fold_range:>8.2f}  [{s.sd_ci[0]:,.4g}, {s.sd_ci[1]:,.4g}]{flag}"
            )
        cache = [r.get("cache_read_input_tokens", 0) for r in rs]
        lines.append(
            f"  cache_read_input_tokens: min={min(cache):,} max={max(cache):,} "
            f"-- a warm cache is a cost difference that has nothing to do with the index"
        )
        lines.append("")
    lines.append(
        "* wall_s is the noisiest metric and has a floor effect on small repos "
        "(codegraph's own README concedes this). Weigh it last."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------
# 2. Sizing
# ---------------------------------------------------------------------------------


def effective_n(n_tasks: int, reps: int, icc: float) -> float:
    """`m*q / (1 + (q-1)*ICC)` -- what `n_tasks` tasks re-run `reps` times are worth.

    The expression saturates at `n_tasks / icc` however large `reps` grows, which is
    the whole point and the thing a per-run count hides. Repeats of one task share its
    prompt, its repo and its difficulty; past a handful they stop adding evidence and
    only sharpen the estimate of that one task's mean.
    """
    if icc <= 0:
        return float(n_tasks * reps)
    return (n_tasks * reps) / (1 + (reps - 1) * icc)


def sizing_report(
    cv: float,
    icc: float = 0.05,
    budget_runs: int | None = None,
    n_arms: int = 3,
    n_tasks: int | None = None,
) -> str:
    """What a budget buys, and the smallest ratio it can resolve.

    Works on `log(metric)`, where a k-fold difference becomes a difference of `log k`
    and the sd of the log is approximately the coefficient of variation. That turns
    `ablation.required_n` -- written for bounded scores -- into the right tool for a
    ratio without reimplementing it.

    The `n` columns are EFFECTIVE observations, not runs, and the two are not close.
    Runs are clustered inside tasks, so a budget spent on more repeats of the same few
    tasks converts to effective n at a punishing rate; `effective_n` does that
    conversion and the BUDGET block shows both numbers side by side. Sizing a design
    against its raw run count is the specific mistake that lets four runs per arm look
    like a measurement.
    """
    lines = [
        f"SIZING at cv = {cv:.1%} (sd of log(metric) ~ {cv:.3f}), assumed ICC = {icc:.3f}",
        "  n below is EFFECTIVE observations. Convert to runs with the BUDGET block.",
        "",
        f"  {'ratio to detect':>16} {'eff. n @50% power':>18} {'eff. n @80% power':>18}",
    ]
    for ratio in (1.1, 1.25, 1.5, 2.0, 3.0):
        delta = math.log(ratio)
        lines.append(
            f"  {ratio:>15.2f}x {required_n(cv, delta, 0.50):>18,} "
            f"{required_n(cv, delta, 0.80):>18,}"
        )
    lines.append("")
    lines.append("  half-width of the log-ratio CI by effective n (1.00x = no effect):")
    for n in (4, 8, 16, 32, 64, 128, 256):
        hw = ci_half_width(cv, n)
        lines.append(f"    n={n:<4} +/- {hw:.3f} in log  =>  interval spans "
                     f"{math.exp(-hw):.2f}x - {math.exp(hw):.2f}x")

    if budget_runs:
        per_arm = budget_runs // n_arms
        lines += ["", f"  BUDGET: {budget_runs} runs / {n_arms} arms = {per_arm} per arm.",
                  f"  {'tasks':>7} {'reps':>6} {'runs/arm':>9} {'effective n':>12} "
                  f"{'resolves @50%':>14} {'@80%':>8}"]
        for tasks in (n_tasks,) if n_tasks else (4, 8, 12, 20, 30):
            reps = max(per_arm // tasks, 1)
            eff = effective_n(tasks, reps, icc)
            hw = ci_half_width(cv, max(eff, 1.0))
            lines.append(
                f"  {tasks:>7} {reps:>6} {tasks * reps:>9} {eff:>12.0f} "
                f"{math.exp(hw):>13.2f}x {math.exp(hw * 1.43):>7.2f}x"
            )
        cap = (n_tasks or 12) / icc if icc > 0 else float("inf")
        lines.append(
            f"  CEILING: with {n_tasks or 12} tasks, effective n cannot exceed "
            f"{cap:.0f} however many times each is re-run. Add TASKS, not reps."
        )

    lines += [
        "",
        f"  CALIBRATION FLOOR: {CALIBRATION_FLOOR}. Below this the percentile bootstrap "
        f"over-rejects -- measured in this repo at 11.9% where it claimed 5% -- so an n "
        f"large enough to RESOLVE an effect can still be too small for its interval to "
        f"mean what it says. Read a sub-{CALIBRATION_FLOOR} interval as descriptive.",
        "  The ICC above is an ASSUMPTION until there are enough tasks to estimate it; "
        "`analyze.py` reports the measured design effect once the matrix has run.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------
# 3. Paired comparison, clustered on task
# ---------------------------------------------------------------------------------


def _cluster_draws(labels: Sequence[str], resamples: int, seed: int) -> list[list[int]]:
    """Bootstrap draws whose unit is the TASK. Mirrors `ablation._resample_clusters`.

    Not imported because that function resamples over a `Scorecard`'s per-query rows,
    and the unit here is a per-task paired ratio rather than a graded query. The
    resampling scheme is identical and deliberately so.
    """
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    keys = sorted(groups)
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    if len(keys) <= 1:
        n = len(labels)
        return [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]
    draws = []
    for _ in range(resamples):
        picked: list[int] = []
        for _ in range(len(keys)):
            picked.extend(groups[keys[rng.randrange(len(keys))]])
        draws.append(picked)
    return draws


@dataclass
class Delta:
    """A paired arm-vs-arm ratio with a task-clustered bootstrap interval."""

    metric: str
    arm: str
    baseline: str
    n_tasks: int
    n_pairs: int
    #: Geometric mean of the per-task ratio `arm / baseline`. Below 1.0 the arm is
    #: cheaper. Geometric because the quantity is a ratio: the mean of 0.5x and 2.0x
    #: is 1.0x, not 1.25x.
    ratio: float
    ci_low: float
    ci_high: float
    #: Whether the interval excludes 1.0. Reported ALONGSIDE the calibration warning,
    #: never as a standalone verdict.
    excludes_null: bool
    design_effect: float
    below_calibration_floor: bool


def delta_ci(runs: list[dict], arm: str, baseline: str, metric: str,
             resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> Delta:
    """Paired ratio of `arm` to `baseline`, bootstrapped over TASKS.

    Pairing is by task: each task contributes the ratio of its arm mean to its baseline
    mean, so a task that is simply expensive in every arm cannot swamp the comparison.
    Clustering is also by task, because the repeats inside one task are not independent
    evidence -- they share a prompt, a repo and a difficulty.
    """
    per_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["arm"] in (arm, baseline):
            per_task[r["task_id"]][r["arm"]].append(_value(r, metric))

    tasks, ratios, npairs = [], [], 0
    for tid, byarm in sorted(per_task.items()):
        a, b = byarm.get(arm), byarm.get(baseline)
        if not a or not b:
            continue
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        if ma <= 0 or mb <= 0:
            continue
        tasks.append(tid)
        ratios.append(math.log(ma / mb))
        npairs += min(len(a), len(b))

    if not ratios:
        return Delta(metric, arm, baseline, 0, 0, float("nan"), float("nan"),
                     float("nan"), False, 1.0, True)

    deff = design_effect(ratios, tasks) if len(set(tasks)) > 1 else 1.0
    draws = _cluster_draws(tasks, resamples, seed)
    boots = sorted(statistics.fmean([ratios[i] for i in d]) for d in draws)
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(int(0.975 * len(boots)), len(boots) - 1)]
    point = statistics.fmean(ratios)
    return Delta(
        metric=metric,
        arm=arm,
        baseline=baseline,
        n_tasks=len(tasks),
        n_pairs=npairs,
        ratio=math.exp(point),
        ci_low=math.exp(lo),
        ci_high=math.exp(hi),
        excludes_null=not (lo <= 0.0 <= hi),
        design_effect=deff,
        below_calibration_floor=npairs < CALIBRATION_FLOOR,
    )


def diagnostics(runs: list[dict], all_runs: list[dict]) -> list[str]:
    """Everything that would make the numbers above mean less than they look.

    Printed on the face of the report rather than in a footnote. Each line is a
    condition that a benchmark can satisfy and still be wrong, and the comparison this
    exists to answer trips several of them.
    """
    out = []
    tasks = {r["task_id"] for r in runs}
    if len(tasks) < 5:
        out.append(
            f"ONLY {len(tasks)} TASK(S). The cluster bootstrap resamples tasks, so it "
            f"has {len(tasks)} effective observations however many runs sit inside "
            f"them. Per-task rows are the honest reporting; the pooled interval is a "
            f"placeholder until there are ~5+ tasks."
        )
    void = [r for r in all_runs if not r.get("ok")]
    if void:
        by_arm: dict[str, int] = defaultdict(int)
        for r in void:
            by_arm[r["arm"]] += 1
        out.append(
            f"{len(void)} VOID run(s) excluded {dict(by_arm)}. A void run is one whose "
            f"MCP server never started; it is not a cheap success."
        )
    unused = [r for r in runs
              if r["arm"] != "bare" and r.get("index_offered") and not r.get("index_tool_calls")]
    if unused:
        by_arm2: dict[str, int] = defaultdict(int)
        for r in unused:
            by_arm2[r["arm"]] += 1
        out.append(
            f"{len(unused)} run(s) had the index available and NEVER CALLED IT "
            f"{dict(by_arm2)}. Those runs measure the built-in tools, not the index; "
            f"they belong in the average but not in a claim about the index."
        )
    errs = sum(r.get("index_tool_errors", 0) for r in runs)
    if errs:
        out.append(f"{errs} MCP tool call(s) returned an ERROR. A denied or failing "
                   f"call is indistinguishable, in the usage numbers, from an index "
                   f"that did not help.")
    defer = {r["arm"] for r in runs if r.get("index_offered") and not r.get("index_first_class")}
    if defer:
        out.append(
            f"Tool deferral is asymmetric: {sorted(defer)} had tools in the DEFERRED "
            f"pool, costing extra ToolSearch calls for server startup time rather than "
            f"for the index. Compare `n_tool_calls_adj`, or re-run with --defer-parity."
        )
    pos = defaultdict(list)
    for r in runs:
        pos[r.get("arm_position", 0)].append(r.get("cache_read_input_tokens", 0))
    if len(pos) > 1:
        means = {p: statistics.fmean(v) for p, v in sorted(pos.items())}
        spread_ratio = (max(means.values()) + 1) / (min(means.values()) + 1)
        out.append(
            f"Cache-read tokens by arm position: "
            f"{ {p: round(v) for p, v in means.items()} } ({spread_ratio:.2f}x spread). "
            f"Arm order is counterbalanced, so this should not track the arm; if it "
            f"does, position is confounded with arm."
        )
    return out


def format_report(runs: list[dict], all_runs: list[dict], baseline: str = "bare") -> str:
    arms = sorted({r["arm"] for r in runs} - {baseline})
    lines = [
        f"PAIRED DELTAS vs `{baseline}` -- ratio < 1.00 means the arm is cheaper",
        "Pairing and bootstrap clustering are both on TASK.",
        "",
        f"  {'metric':<18} {'arm':<13} {'tasks':>5} {'runs':>5} "
        f"{'ratio':>8} {'95% CI (task-clustered)':>26} {'deff':>6}",
    ]
    for metric in (*METRICS, *ADJUSTED):
        for arm in arms:
            d = delta_ci(runs, arm, baseline, metric)
            if d.n_tasks == 0:
                continue
            mark = "*" if metric in NOISY else " "
            floor = " [below calib. floor]" if d.below_calibration_floor else ""
            lines.append(
                f"{mark} {metric:<18} {arm:<13} {d.n_tasks:>5} {d.n_pairs:>5} "
                f"{d.ratio:>8.3f} {f'[{d.ci_low:.3f}, {d.ci_high:.3f}]':>26} "
                f"{d.design_effect:>6.2f}{floor}"
            )
    diags = diagnostics(runs, all_runs)
    if diags:
        lines += ["", "DIAGNOSTICS -- read these before the table above:"]
        lines += [f"  - {d}" for d in diags]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--baseline", default="bare")
    ap.add_argument("--variance", action="store_true", help="variance report only")
    ap.add_argument("--task", default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--sizing-cv", type=float, default=None,
                    help="print the sizing table for this coefficient of variation")
    ap.add_argument("--budget-runs", type=int, default=None)
    ap.add_argument("--icc", type=float, default=0.05,
                    help="intra-task correlation; drives how many reps are worth buying")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.sizing_cv is not None:
        print(sizing_report(args.sizing_cv, icc=args.icc,
                            budget_runs=args.budget_runs, n_tasks=args.n_tasks))
        return 0

    all_runs = load(args.results_dir, include_void=True)
    runs = [r for r in all_runs if r.get("ok")]
    if not runs:
        print(f"no usable runs in {args.results_dir}")
        return 1

    if args.variance:
        print(variance_report(runs, args.task, args.arm))
        return 0

    if args.json:
        arms = sorted({r["arm"] for r in runs} - {args.baseline})
        print(json.dumps(
            {
                "n_runs": len(runs),
                "n_void": len(all_runs) - len(runs),
                "deltas": [
                    vars(delta_ci(runs, a, args.baseline, m))
                    for m in (*METRICS, *ADJUSTED) for a in arms
                ],
                "diagnostics": diagnostics(runs, all_runs),
            },
            indent=2, default=str,
        ))
        return 0

    print(variance_report(runs, args.task, args.arm))
    print()
    print(format_report(runs, all_runs, args.baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
