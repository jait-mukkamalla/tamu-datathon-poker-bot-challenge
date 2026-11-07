"""
Offline builder for data/preflop_equity.npy (run from the repo root, Python 3.12):

    python tools/build_preflop.py [--samples 4000] [--workers 20]

Output: uint16 array of length 376,992, indexed by eqlib.combo_index(hole);
value / 65535 = Monte Carlo equity (win + tie/2) of that 5-card hole hand vs
one uniformly random 5-card hand over a random 5-card board. Suit-isomorphic
hands share one computed value.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import combinations, permutations
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eqlib import PREFLOP_PATH, combo_index_arr, omaha_scores  # noqa: E402

BATCH = 1000


def all_combos_by_index() -> np.ndarray:
    """(376992, 5) array whose row i is the hand with eqlib.combo_index == i."""
    combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    out = np.empty_like(combos)
    out[combo_index_arr(combos)] = combos
    return out


def canonical_classes() -> tuple[np.ndarray, np.ndarray]:
    """Return (canon, reps): canon[i] = index of the class representative of
    combo i; reps = sorted array of unique representative indices."""
    combos = all_combos_by_index()
    best = np.arange(len(combos), dtype=np.int64)
    ranks, suits = combos >> 2, combos & 3
    for perm in permutations(range(4)):
        mapped = np.sort(ranks * 4 + np.array(perm)[suits], axis=1)
        best = np.minimum(best, combo_index_arr(mapped))
    return best, np.unique(best)


def equity_of(hole: np.ndarray, n: int, rng: np.random.Generator) -> float:
    rest = np.setdiff1d(np.arange(36), hole)
    total = 0.0
    done = 0
    while done < n:
        m = min(BATCH, n - done)
        draw = rest[np.argsort(rng.random((m, 31)), axis=1)[:, :10]]
        mine = np.broadcast_to(hole, (m, 5))
        a = omaha_scores(mine, draw[:, 5:])
        b = omaha_scores(draw[:, :5], draw[:, 5:])
        total += (a > b).sum() + 0.5 * (a == b).sum()
        done += m
    return total / n


def work(args):
    idxs, holes, n, seed = args
    rng = np.random.default_rng(seed)
    return idxs, [equity_of(h, n, rng) for h in holes]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="only first N classes (timing test)")
    ap.add_argument("--out", default=PREFLOP_PATH)
    a = ap.parse_args()

    t0 = time.time()
    canon, reps = canonical_classes()
    print(f"{len(reps)} suit-isomorphic classes ({time.time() - t0:.1f}s)", flush=True)
    all_combos = all_combos_by_index()
    todo = reps[: a.limit] if a.limit else reps
    chunks = []
    step = 50
    for s in range(0, len(todo), step):
        ids = todo[s:s + step]
        chunks.append((ids, [all_combos[i] for i in ids], a.samples, 1234 + s))

    eq_by_rep = {}
    with Pool(a.workers) as pool:
        for k, (ids, eqs) in enumerate(pool.imap_unordered(work, chunks)):
            for i, e in zip(ids, eqs):
                eq_by_rep[int(i)] = e
            if k % 20 == 0:
                print(f"  {len(eq_by_rep)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)

    if a.limit:
        print("limit mode: not writing output")
        return
    table = np.zeros(len(canon), dtype=np.uint16)
    lut = np.zeros(len(canon), dtype=np.float64)
    for r, e in eq_by_rep.items():
        lut[r] = e
    table[:] = np.rint(lut[canon] * 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.save(a.out, table)
    print(f"wrote {a.out} ({table.nbytes / 1e6:.2f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
