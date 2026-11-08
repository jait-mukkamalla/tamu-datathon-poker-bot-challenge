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

Early stopping (on by default; --no-sprt turns it off): both stages play their seeds
in batches of --batch-seeds seeds x every pool opponent and, after each batch, look at
the paired differences so far (n matches, mean gain g, sample variance s2, se). A Wald
SPRT of H0: gain = 0 against H1: gain = SPRT_DELTA uses the normal log-likelihood ratio

    LLR = n * SPRT_DELTA / s2 * (g - SPRT_DELTA / 2)

  - early REJECT (both stages) when LLR < SPRT_LOWER = log(beta / (1 - alpha)), or when
    g < -FUTILITY_T * se. This is where the savings are: most candidates are no better
    than the current best. The SPRT bound bites when the matches are precise (se small
    next to SPRT_DELTA); the futility bound when they are noisy (being 1 se behind
    after a third of the seeds and still finishing > SCREEN_T se ahead is a ~3% event).
  - screen: early PASS (to the confirm stage, which still has to be passed) when
    LLR > SPRT_UPPER = log((1 - beta) / alpha) and g > SCREEN_T * se.
  - confirm: early KEEP only if very strong: at least half the confirm seeds played,
    g > MIN_GAIN and g > EARLY_ACCEPT_T (3, vs ACCEPT_T = 2) standard errors.
  - otherwise the last batch completes all seeds and the fixed-sample rules above decide.
No early decision is taken on fewer than SPRT_MIN_N paired matches. Seeds are allocated
exactly as without early stopping (a confirmation always uses up --confirm-seeds fresh
seeds), so the same seeds give the same decisions run after run, and every result is
cached by (params, opponent, seed), so nothing is played twice: the best's round
results are reused by every screen, and a kept candidate's screen results become the
new baseline (only its unplayed round seeds are filled in).

Run with a fixed sample cap so parallel load does not change the bots' strength:

    BOT_FIXED_CAP=1500 python tools/tune.py --pool DIR1,DIR2,... --rounds 1
    ... --batch-seeds 2     # smaller batches: earlier stops, but less parallelism
    ... --no-sprt           # old behaviour: every candidate plays every seed

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
# early stopping (see the docstring)
SPRT_DELTA = 0.075               # H1: gain of this many bb/deal
SPRT_ALPHA, SPRT_BETA = 0.05, 0.20
SPRT_UPPER = math.log((1 - SPRT_BETA) / SPRT_ALPHA)   # +2.77
SPRT_LOWER = math.log(SPRT_BETA / (1 - SPRT_ALPHA))   # -1.56
SPRT_MIN_N = 6                   # no early decision on fewer paired matches
EARLY_ACCEPT_T = 3.0             # confirm: an early keep needs this many std errors
FUTILITY_T = 1.0                 # early reject also when behind by this many std errors

_CACHE: dict[tuple, float] = {}  # (params json, opponent, seed, deals) -> mean bb/deal
PLAYED = [0, 0]                  # [matches played, matches without early stopping/cache]


def _key(params: dict, opp: str, seed: int, deals: int) -> tuple:
    return json.dumps(params, sort_keys=True), opp, seed, deals


def evaluate_many(me: str, groups: list[tuple[dict, list[str], list[int]]], deals: int,
                  workers: int) -> list[list[float]]:
    """For each (params, pool, seeds) group, the mean bb/deal of each (opponent, seed)
    match in pool-major order. All uncached matches run together in one parallel map."""
    todo: dict[tuple, tuple] = {}
    for params, pool, seeds in groups:
        for opp in pool:
            for s in seeds:
                k = _key(params, opp, s, deals)
                if k not in _CACHE and k not in todo:
                    todo[k] = ((me, params), (opp, None), s, deals)
    if todo:
        for k, r in zip(todo, run_matches(list(todo.values()), workers)):
            _CACHE[k] = r["mean"]
        PLAYED[0] += len(todo)
    return [[_CACHE[_key(p, opp, s, deals)] for opp in pool for s in seeds]
            for p, pool, seeds in groups]


def evaluate(me: str, params: dict, pool: list[str], seeds: list[int], deals: int,
             workers: int) -> list[float]:
    """Mean bb/deal of each (opponent, seed) match, in pool-major order."""
    return evaluate_many(me, [(params, pool, seeds)], deals, workers)[0]


def paired(cand: list[float], best: list[float]) -> tuple[float, float]:
    d = [c - b for c, b in zip(cand, best)]
    return sum(d) / len(d), statistics.stdev(d) / math.sqrt(len(d))


def sprt_llr(cand: list[float], best: list[float]) -> float:
    """Normal-approximation log-likelihood ratio of gain = SPRT_DELTA vs gain = 0."""
    d = [c - b for c, b in zip(cand, best)]
    g = sum(d) / len(d)
    var = max(statistics.variance(d), 1e-12)
    return len(d) * SPRT_DELTA / var * (g - SPRT_DELTA / 2)


def batches(seeds: list[int], size: int) -> list[list[int]]:
    """Growing seed prefixes, one per batch; the last one is all the seeds."""
    return [seeds[:i] for i in range(size, len(seeds), size)] + [seeds]


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


def screen(me, cand, base, pool, seeds, deals, workers, sprt, bs):
    """Candidate vs the best's known results `base` on the round seeds.
    Returns (passed, gain, se, note)."""
    parts = batches(seeds, bs) if sprt else [seeds]
    for part in parts:
        res = evaluate(me, cand, pool, part, deals, workers)
        ref = [base[i * len(seeds) + j] for i in range(len(pool)) for j in range(len(part))]
        if part is parts[-1] or len(res) < SPRT_MIN_N:
            continue
        gain, se = paired(res, ref)
        llr = sprt_llr(res, ref)
        if llr < SPRT_LOWER or gain < -FUTILITY_T * se:
            return False, gain, se, f" [early stop {len(part)}/{len(seeds)} seeds]"
        if llr > SPRT_UPPER and gain > SCREEN_T * se:
            return True, gain, se, f" [early pass {len(part)}/{len(seeds)} seeds]"
    gain, se = paired(res, ref)
    return gain > SCREEN_T * se, gain, se, ""


def confirm(me, cand, best, pool, cseeds, deals, workers, sprt, bs):
    """Candidate vs best, both played on fresh seeds.
    Returns (keep, gain, se, note, c_new, c_old, seeds played)."""
    parts = batches(cseeds, bs) if sprt else [cseeds]
    for part in parts:
        c_new, c_old = evaluate_many(me, [(cand, pool, part), (best, pool, part)],
                                     deals, workers)
        if part is parts[-1]:
            break
        if len(c_new) < SPRT_MIN_N:
            continue
        cg, cse = paired(c_new, c_old)
        note = f" [early {len(part)}/{len(cseeds)} seeds]"
        if sprt_llr(c_new, c_old) < SPRT_LOWER or cg < -FUTILITY_T * cse:
            return False, cg, cse, note, c_new, c_old, len(part)
        if 2 * len(part) >= len(cseeds) and cg > MIN_GAIN and cg > EARLY_ACCEPT_T * cse:
            return True, cg, cse, note, c_new, c_old, len(part)
    cg, cse = paired(c_new, c_old)
    keep = cg > MIN_GAIN and cg > ACCEPT_T * cse
    return keep, cg, cse, "", c_new, c_old, len(cseeds)


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
    ap.add_argument("--batch-seeds", type=int, default=3,
                    help="early stopping: seeds per batch (each batch plays every opponent)")
    ap.add_argument("--no-sprt", action="store_true",
                    help="disable early stopping: every candidate plays every seed")
    args = ap.parse_args()
    pool = args.pool.split(",")
    best = json.loads(args.start)
    defaults = load_defaults(args.me)
    keys = args.params.split(",") if args.params else list(STEPS)
    sprt, bs = not args.no_sprt, max(1, args.batch_seeds)
    confirm_next = 20000
    t_start = time.time()

    def val(k):
        return best.get(k, defaults[k])

    def save():
        with open(args.out, "w") as f:
            json.dump(best, f, indent=1, sort_keys=True)

    for rnd in range(args.rounds):
        scale = 0.5 ** rnd
        seeds = list(range(10000 + 1000 * rnd, 10000 + 1000 * rnd + args.seeds_per_round))
        t0 = time.time()
        PLAYED[1] += len(pool) * len(seeds)
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
                PLAYED[1] += len(pool) * len(seeds)
                passed, gain, se, note = screen(args.me, cand, base, pool, seeds, args.deals,
                                                args.workers, sprt, bs)
                line = (f"  {k:16s} {val(k):+.3f} -> {cand[k]:+.3f}: {gain:+.3f} "
                        f"(se {se:.3f}){note}")
                if not passed:
                    print(line, flush=True)
                    continue
                cseeds = list(range(confirm_next, confirm_next + args.confirm_seeds))
                confirm_next += args.confirm_seeds
                PLAYED[1] += 2 * len(pool) * len(cseeds)
                keep, cg, cse, cnote, c_new, c_old, n = confirm(
                    args.me, cand, best, pool, cseeds, args.deals, args.workers, sprt, bs)
                print(f"{line} | confirm {cg:+.3f} (se {cse:.3f})"
                      f"{'  KEEP' if keep else '  rejected'}{cnote}"
                      f"  [{per_opponent(c_new, c_old, pool, n)}]", flush=True)
                if keep:
                    # the new best needs results on every round seed (played ones are cached)
                    best = cand
                    base = evaluate(args.me, best, pool, seeds, args.deals, args.workers)
                    save()
                    break  # this direction worked: skip the opposite one
        print(f"after round {rnd}: {json.dumps(best, sort_keys=True)}", flush=True)
    save()
    print(f"matches played: {PLAYED[0]} (without early stopping: {PLAYED[1]}); "
          f"{time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
