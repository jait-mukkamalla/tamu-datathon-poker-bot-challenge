"""
tools/tune.py - coordinate-search tuning of agent.py's P dict (LOCAL ONLY).

Each candidate plays duplicate matches against an opponent pool on a set of seeds
shared by every candidate in the round (common random numbers: only the parameter
changes between runs). A change is kept only if its paired improvement over the
current best is positive by > ACCEPT_T standard errors and > MIN_GAIN bb/deal.
Seeds are fresh each round so a lucky seed set cannot keep a bad change.

    python tools/tune.py --pool DIR1,DIR2,... --rounds 2 --seeds-per-round 10
Writes progress to stdout and the best params as JSON to --out.
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
}
ACCEPT_T, MIN_GAIN = 1.5, 0.05


def evaluate(me: str, params: dict, pool: list[str], seeds: list[int], deals: int, workers: int):
    jobs = [((me, params), (opp, None), s, deals) for opp in pool for s in seeds]
    return [r["mean"] for r in run_matches(jobs, workers)]


def paired(cand: list[float], best: list[float]) -> tuple[float, float]:
    d = [c - b for c, b in zip(cand, best)]
    return sum(d) / len(d), statistics.stdev(d) / math.sqrt(len(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", default=".")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seeds-per-round", type=int, default=10)
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--start", default="{}")
    ap.add_argument("--out", default="tuned_params.json")
    args = ap.parse_args()
    pool = args.pool.split(",")
    best = json.loads(args.start)
    with open(os.path.join(args.me, "agent.py"), encoding="utf-8") as f:
        src = f.read()
    # read P's defaults without starting a server: exec just the dict literal
    start = src.index("P = {")
    end = src.index("}\n", start) + 1
    ns: dict = {}
    exec(src[start:end], ns)
    defaults = ns["P"]

    def val(k):
        return best.get(k, defaults[k])

    for rnd in range(args.rounds):
        scale = 1.0 if rnd == 0 else 0.5 ** rnd
        seeds = list(range(10000 + 1000 * rnd, 10000 + 1000 * rnd + args.seeds_per_round))
        t0 = time.time()
        base = evaluate(args.me, best, pool, seeds, args.deals, args.workers)
        print(f"round {rnd}: best mean {sum(base) / len(base):+.3f} over {len(base)} matches "
              f"({time.time() - t0:.0f}s)", flush=True)
        for k, step in STEPS.items():
            for sign in (+1, -1):
                cand = dict(best)
                cand[k] = round(val(k) + sign * step * scale, 4)
                res = evaluate(args.me, cand, pool, seeds, args.deals, args.workers)
                gain, se = paired(res, base)
                keep = gain > MIN_GAIN and gain > ACCEPT_T * se
                print(f"  {k:16s} {val(k):+.3f} -> {cand[k]:+.3f}: {gain:+.3f} (se {se:.3f})"
                      f"{'  KEEP' if keep else ''}", flush=True)
                if keep:
                    best, base = cand, res
                    with open(args.out, "w") as f:
                        json.dump(best, f, indent=1, sort_keys=True)
                    break  # this direction worked: skip the opposite one
        print(f"after round {rnd}: {json.dumps(best, sort_keys=True)}", flush=True)
    with open(args.out, "w") as f:
        json.dump(best, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
