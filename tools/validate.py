"""
tools/validate.py - per-opponent A/B of a candidate bot vs a reference bot (LOCAL ONLY).

Both bots play the same duplicate matches (same opponents, same held-out seeds), so
the per-opponent difference is paired and its standard error is small. Any opponent
the candidate does clearly worse against (diff < -2 se) is flagged WORSE.

    python tools/validate.py                          # . vs ../poker-opps/ref
    python tools/validate.py --cand DIR --cand-params '{"bluff_on": 0}'
    python tools/validate.py --pool old,tools/opponents/nit.py --seeds 501-560

Seeds: keep 500-999 for validation (tune.py uses 10000+ and 20000+).
Default pool: snapshots in ../poker-opps (old, frozen0, rb0), the four baselines,
and every bot in tools/opponents (*.py single-file bots or subdirectories with agent.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import time

os.environ.setdefault("BOT_FIXED_CAP", "1500")  # parallel load must not change bot strength
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import ROOT, _seeds, run_matches  # noqa: E402

OPPS_DIR = os.path.normpath(os.path.join(ROOT, "..", "poker-opps"))
if not os.path.isdir(OPPS_DIR):  # running from a git worktree elsewhere
    OPPS_DIR = r"C:\Users\bbcla\Downloads\poker-opps"


def default_pool() -> list[str]:
    pool = [os.path.join(OPPS_DIR, d) for d in ("old", "frozen0", "rb0")
            if os.path.isdir(os.path.join(OPPS_DIR, d))]
    pool += sorted(glob.glob(os.path.join(ROOT, "baseline-bots", "baseline_*.py")))
    od = os.path.join(ROOT, "tools", "opponents")
    for p in sorted(glob.glob(os.path.join(od, "*"))):
        if (p.endswith(".py") and not os.path.basename(p).startswith("_")) or \
                os.path.isfile(os.path.join(p, "agent.py")):
            pool.append(p)
    return pool


def resolve(name: str) -> str:
    """Accept a path, or a short name of a ../poker-opps snapshot or tools/opponents bot."""
    for cand in (name, os.path.join(OPPS_DIR, name), os.path.join(ROOT, "tools", "opponents", name),
                 os.path.join(ROOT, "tools", "opponents", name + ".py"),
                 os.path.join(ROOT, "baseline-bots", name), os.path.join(ROOT, "baseline-bots", name + ".py")):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise SystemExit(f"opponent not found: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default=ROOT)
    ap.add_argument("--cand-params", default=None)
    ap.add_argument("--base", default=os.path.join(OPPS_DIR, "ref"))
    ap.add_argument("--base-params", default=None)
    ap.add_argument("--pool", default=None, help="comma-separated opponents (default: full pool)")
    ap.add_argument("--seeds", default="501-530")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", default=None, help="also write results to this JSON file")
    args = ap.parse_args()
    cp = json.loads(args.cand_params) if args.cand_params else None
    bp = json.loads(args.base_params) if args.base_params else None
    pool = [resolve(p) for p in args.pool.split(",")] if args.pool else default_pool()
    seeds = _seeds(args.seeds)
    print(f"cand {args.cand} {cp or ''}\nbase {args.base} {bp or ''}\n"
          f"{len(pool)} opponents x {len(seeds)} seeds x {args.deals} deals", flush=True)
    t0 = time.time()
    jobs = []
    for opp in pool:
        jobs += [((args.cand, cp), (opp, None), s, args.deals) for s in seeds]
        jobs += [((args.base, bp), (opp, None), s, args.deals) for s in seeds]
    res = run_matches(jobs, args.workers)
    n, out, worse = len(seeds), [], 0
    print(f"{'opponent':30s} {'cand':>8s} {'base':>8s} {'diff':>8s} {'se':>6s}  errors(c)")
    for i, opp in enumerate(pool):
        block = res[2 * n * i: 2 * n * (i + 1)]
        c, b = [r["mean"] for r in block[:n]], [r["mean"] for r in block[n:]]
        d = [x - y for x, y in zip(c, b)]
        md = sum(d) / n
        se = statistics.stdev(d) / math.sqrt(n) if n > 1 else float("nan")
        errs = sum(r["errors"][0] for r in block[:n])
        flag = "  WORSE" if md < -2 * se else ("  better" if md > 2 * se else "")
        worse += flag == "  WORSE"
        name = os.path.basename(opp.rstrip("/\\"))
        print(f"{name:30s} {sum(c) / n:+8.3f} {sum(b) / n:+8.3f} {md:+8.3f} {se:6.3f}  {errs}{flag}",
              flush=True)
        out.append({"opp": name, "cand": sum(c) / n, "base": sum(b) / n, "diff": md, "se": se,
                    "cand_errors": errs})
    avg = sum(o["diff"] for o in out) / len(out)
    print(f"AVERAGE diff {avg:+.3f} bb/deal; {worse} opponent(s) WORSE; {time.time() - t0:.0f}s")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"cand": args.cand, "cand_params": cp, "base": args.base, "base_params": bp,
                       "seeds": args.seeds, "results": out}, f, indent=1)


if __name__ == "__main__":
    main()
