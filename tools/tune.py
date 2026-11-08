"""
tools/tune.py - coordinate-search tuning of agent.py's P dict (LOCAL ONLY).

For each parameter the tuner tries one step up and one step down from the current
best. A change has to pass two tests before it is kept:

  1. screen:  on this round's seeds it beats the current best (whose results on
              those seeds are already known) by > SCREEN_T standard errors.
  2. confirm: on FRESH seeds, the candidate and the current best are both played,
              and the candidate wins again by > ACCEPT_T standard errors and
              > MIN_GAIN bb/deal.

The confirmation stage is what makes a kept change trustworthy: the screen alone
reuses one baseline run for every candidate (one unlucky baseline makes everything
look good) and picks the luckiest of many tries. Every match is a duplicate match on
common deals, paired by (opponent, seed); tools/sim.py reseeds the bots so the same
deals replay the same way.

Run with a fixed sample cap so parallel load does not change the bots' strength:

    BOT_FIXED_CAP=1500 python tools/tune.py --pool DIR1,DIR2,... --rounds 1

Seeds: rounds use 10000+, confirmations 20000+; keep validation seeds elsewhere
(for example 500-999). Writes the best params as JSON to --out after every change.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import run_matches  # noqa: E402

STEPS = {
    "open_pot": 0.04, "open_half": 0.04, "raise_pot": 0.04, "raise_half": 0.04,
    "pre_open_pot": 0.04, "pre_open_half": 0.04, "pre_raise_pot": 0.04, "pre_raise_half": 0.04,
    "call_margin": 0.02, "shade_scale": 0.05, "shade_per_raise": 0.02,
    "off_flop": 0.03, "off_turn": 0.03, "off_river": 0.03, "off_ip": 0.03,
    "pre_off_btn": 0.03, "spr_coef": 0.03, "call_off_river": 0.02,
    "jitter": 0.02, "range_pow": 0.25, "bluff_margin": 0.03,
}
# allowed ranges; a step outside is skipped (e.g. a negative jitter does nothing)
BOUNDS = {
    "jitter": (0.0, 0.2), "range_pow": (0.0, 3.0), "shade_scale": (0.3, 1.2),
    "shade_per_raise": (0.0, 0.3), "call_margin": (-0.1, 0.3),
}
SCREEN_T = 1.0                   # screen: candidate must lead by > this many std errors
ACCEPT_T, MIN_GAIN = 2.0, 0.03   # confirm: lead on fresh seeds by both of these


def evaluate(me: str, params: dict, pool: list[str], seeds: list[int], deals: int,
             workers: int) -> list[float]:
    """Mean bb/deal of each (opponent, seed) match, in pool-major order."""
    jobs = [((me, params), (opp, None), s, deals) for opp in pool for s in seeds]
    return [r["mean"] for r in run_matches(jobs, workers)]


def paired(cand: list[float], best: list[float]) -> tuple[float, float]:
    d = [c - b for c, b in zip(cand, best)]
    return sum(d) / len(d), statistics.stdev(d) / math.sqrt(len(d))


def per_opponent(cand: list[float], best: list[float], pool: list[str], n: int) -> str:
    """Gain per opponent, e.g. 'old:+0.12 baseline_random.py:-0.03'."""
    parts = []
    for i, opp in enumerate(pool):
        name = os.path.basename(opp.rstrip("/").rstrip("\\"))
        c, b = cand[i * n:(i + 1) * n], best[i * n:(i + 1) * n]
        parts.append(f"{name}:{(sum(c) - sum(b)) / n:+.2f}")
    return " ".join(parts)


def load_defaults(me: str) -> dict:
    """P's defaults from me/agent.py, without starting a server: exec the dict literal."""
    with open(os.path.join(me, "agent.py"), encoding="utf-8") as f:
        src = f.read()
    start = src.index("P = {")
    end = src.index("}\n", start) + 1
    ns: dict = {}
    exec(src[start:end], ns)
    return ns["P"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", default=".")
    ap.add_argument("--pool", required=True, help="comma-separated opponent bot dirs/files")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seeds-per-round", type=int, default=10)
    ap.add_argument("--confirm-seeds", type=int, default=10)
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--params", default=None,
                    help="comma-separated subset of STEPS to tune (default: all)")
    ap.add_argument("--start", default="{}", help="JSON params to start from")
    ap.add_argument("--from-param", default=None, help="resume round 0 at this parameter")
    ap.add_argument("--out", default="tuned_params.json")
    args = ap.parse_args()
    pool = args.pool.split(",")
    best = json.loads(args.start)
    defaults = load_defaults(args.me)
    keys = args.params.split(",") if args.params else list(STEPS)
    confirm_next = 20000

    def val(k):
        return best.get(k, defaults[k])

    def save():
        with open(args.out, "w") as f:
            json.dump(best, f, indent=1, sort_keys=True)

    for rnd in range(args.rounds):
        scale = 0.5 ** rnd
        seeds = list(range(10000 + 1000 * rnd, 10000 + 1000 * rnd + args.seeds_per_round))
        t0 = time.time()
        base = evaluate(args.me, best, pool, seeds, args.deals, args.workers)
        print(f"round {rnd}: best mean {sum(base) / len(base):+.3f} over {len(base)} matches "
              f"({time.time() - t0:.0f}s)", flush=True)
        todo = keys
        if rnd == 0 and args.from_param:
            todo = keys[keys.index(args.from_param):]
        for k in todo:
            lo, hi = BOUNDS.get(k, (-math.inf, math.inf))
            for sign in (+1, -1):
                cand = dict(best)
                cand[k] = round(val(k) + sign * STEPS[k] * scale, 4)
                if not lo <= cand[k] <= hi:
                    continue
                res = evaluate(args.me, cand, pool, seeds, args.deals, args.workers)
                gain, se = paired(res, base)
                line = f"  {k:16s} {val(k):+.3f} -> {cand[k]:+.3f}: {gain:+.3f} (se {se:.3f})"
                if not gain > SCREEN_T * se:
                    print(line, flush=True)
                    continue
                cseeds = list(range(confirm_next, confirm_next + args.confirm_seeds))
                confirm_next += args.confirm_seeds
                c_new = evaluate(args.me, cand, pool, cseeds, args.deals, args.workers)
                c_old = evaluate(args.me, best, pool, cseeds, args.deals, args.workers)
                cg, cse = paired(c_new, c_old)
                keep = cg > MIN_GAIN and cg > ACCEPT_T * cse
                print(f"{line} | confirm {cg:+.3f} (se {cse:.3f})"
                      f"{'  KEEP' if keep else '  rejected'}"
                      f"  [{per_opponent(c_new, c_old, pool, len(cseeds))}]", flush=True)
                if keep:
                    best, base = cand, res
                    save()
                    break  # this direction worked: skip the opposite one
        print(f"after round {rnd}: {json.dumps(best, sort_keys=True)}", flush=True)
    save()


if __name__ == "__main__":
    main()
