"""
Offline builder for data/postflop_equity.npy (run from the repo root, Python 3.12):

    python tools/build_postflop.py [--samples 600000] [--workers 20]

Output: float32 array shape (3, 9, 9, 2): [street (flop/turn/river)]
[made-hand category][primary rank] -> (equity vs one random hand, sample count).
"Made hand" = best omaha score on the board so far; primary rank = first
tie-break rank of that score (pair rank, trips rank, flush high card, ...).
Equity counts win + tie/2 against a random opponent hand over the remaining
board cards. Sparse cells are shrunk toward their category mean at lookup.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eqlib import omaha_scores  # noqa: E402

OUT = os.path.join(ROOT, "data", "postflop_equity.npy")
BATCH = 20000


def work(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    sums = np.zeros((3, 9, 9))
    cnts = np.zeros((3, 9, 9))
    for _ in range(n // BATCH):
        d = np.argsort(rng.random((BATCH, 36)), axis=1)[:, :15]
        hole, opp, board = d[:, :5], d[:, 5:10], d[:, 10:]
        fin_me = omaha_scores(hole, board)
        fin_op = omaha_scores(opp, board)
        eq = np.where(fin_me > fin_op, 1.0, np.where(fin_me == fin_op, 0.5, 0.0))
        for st, k in enumerate((3, 4, 5)):
            sc = fin_me if k == 5 else omaha_scores(hole, board[:, :k])
            cat, top = sc >> 20, (sc >> 16) & 15
            np.add.at(sums[st], (cat, top), eq)
            np.add.at(cnts[st], (cat, top), 1)
    return sums, cnts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=600000)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    per = max(BATCH, (a.samples // a.workers) // BATCH * BATCH)
    with Pool(a.workers) as pool:
        res = pool.map(work, [(per, 77 + i) for i in range(a.workers)])
    sums = sum(r[0] for r in res)
    cnts = sum(r[1] for r in res)
    out = np.zeros((3, 9, 9, 2), dtype=np.float32)
    out[..., 0] = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.5)
    out[..., 1] = cnts
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, out)
    print(f"wrote {OUT} from {int(cnts[0].sum())} samples/street in {time.time() - t0:.0f}s")
    names = "high pair 2pair strt trips full flush quad sflush".split()
    for st, sn in enumerate(("flop", "turn", "river")):
        print(sn, {names[c]: round(float((sums[st, c].sum() / max(cnts[st, c].sum(), 1))), 3)
                   for c in range(9) if cnts[st, c].sum() > 0})


if __name__ == "__main__":
    main()
