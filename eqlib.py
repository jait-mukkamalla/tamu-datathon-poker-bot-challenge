"""
eqlib.py - numpy helpers shared by the bot and the offline table builders.

Vectorized Omaha scoring: a 5-card hand's score depends only on its rank
multiset and whether it is a flush, and both can be computed by ADDING
per-card keys (order independent, so no sorting):
    rank key  = 5 ** rank   (sum over 5 cards < 5 * 5**8 ~ 1.95M)
    suit key  = 1 << (3*suit) (flush <=> the sum is 5 << (3*suit))
Two lookup arrays (non-flush / flush) turn the keys into evaluator scores.
"""

from __future__ import annotations

import os
from itertools import combinations
from math import comb

import numpy as np

from evaluator import get_table

_HERE = os.path.dirname(os.path.abspath(__file__))
PREFLOP_PATH = os.path.join(_HERE, "data", "preflop_equity.npy")

CARD_RK = np.array([5 ** (c >> 2) for c in range(36)], dtype=np.int32)
CARD_SK = np.array([1 << (3 * (c & 3)) for c in range(36)], dtype=np.int32)

PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # (10, 2)
TRIPLES = np.array(list(combinations(range(5), 3)), dtype=np.int64)  # (10, 3)

_FLUSH_SUMS = np.zeros(4096, dtype=bool)
for _s in range(4):
    _FLUSH_SUMS[5 << (3 * _s)] = True

_T_NONFLUSH: np.ndarray | None = None
_T_FLUSH: np.ndarray | None = None


def _build_score_arrays() -> None:
    global _T_NONFLUSH, _T_FLUSH
    table = get_table()
    size = 5 ** 9
    nonflush = np.zeros(size, dtype=np.int32)
    flush = np.zeros(size, dtype=np.int32)
    rk = [5 ** (c >> 2) for c in range(36)]
    for combo in combinations(range(36), 5):
        a, b, c, d, e = combo
        key = rk[a] + rk[b] + rk[c] + rk[d] + rk[e]
        score = table[(1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e)]
        if (a & 3) == (b & 3) == (c & 3) == (d & 3) == (e & 3):
            flush[key] = score
        else:
            nonflush[key] = score
    _T_NONFLUSH, _T_FLUSH = nonflush, flush


def score_arrays() -> tuple[np.ndarray, np.ndarray]:
    if _T_NONFLUSH is None:
        _build_score_arrays()
    return _T_NONFLUSH, _T_FLUSH


def omaha_scores(hole: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Vectorized omaha_score. hole, board: int arrays of shape (N, 5).
    Returns int32 scores of shape (N,), identical to evaluator.omaha_score."""
    nonflush, flush = score_arrays()
    hr, hs = CARD_RK[hole], CARD_SK[hole]    # (N, 5)
    br, bs = CARD_RK[board], CARD_SK[board]
    hr2 = hr[:, PAIRS[:, 0]] + hr[:, PAIRS[:, 1]]                            # (N, 10)
    hs2 = hs[:, PAIRS[:, 0]] + hs[:, PAIRS[:, 1]]
    br3 = br[:, TRIPLES[:, 0]] + br[:, TRIPLES[:, 1]] + br[:, TRIPLES[:, 2]]
    bs3 = bs[:, TRIPLES[:, 0]] + bs[:, TRIPLES[:, 1]] + bs[:, TRIPLES[:, 2]]
    rk = hr2[:, :, None] + br3[:, None, :]                                    # (N, 10, 10)
    sk = hs2[:, :, None] + bs3[:, None, :]
    sc = np.where(_FLUSH_SUMS[sk], flush[rk], nonflush[rk])
    return sc.reshape(len(hole), 100).max(axis=1)


# ---------------------------------------------------------------------------
# Combinatorial rank of a 5-card set -> index into a flat 376,992 array
# ---------------------------------------------------------------------------

_COMB = [[comb(n, k) for k in range(6)] for n in range(36)]
COMB_ARR = np.array(_COMB, dtype=np.int64)  # (36, 6)


def combo_index(cards) -> int:
    """Index (0..376991) of a set of 5 distinct cards (any order)."""
    s = sorted(cards)
    return (_COMB[s[0]][1] + _COMB[s[1]][2] + _COMB[s[2]][3]
            + _COMB[s[3]][4] + _COMB[s[4]][5])


def combo_index_arr(sorted_cards: np.ndarray) -> np.ndarray:
    """Vectorized combo_index for an (N, 5) array whose rows are ascending."""
    out = np.zeros(len(sorted_cards), dtype=np.int64)
    for k in range(5):
        out += COMB_ARR[sorted_cards[:, k], k + 1]
    return out


# ---------------------------------------------------------------------------
# Preflop equity lookup (equity vs one random hand, from data/preflop_equity.npy)
# ---------------------------------------------------------------------------

_PREFLOP: np.ndarray | None = None


def preflop_equity(hole) -> float:
    """Approximate equity (0..1) of a 5-card hole hand vs a random hand,
    heads-up to showdown. Falls back to 0.5 if the table file is missing."""
    global _PREFLOP
    if _PREFLOP is None:
        try:
            _PREFLOP = np.load(PREFLOP_PATH)
        except Exception:
            _PREFLOP = np.full(376992, 32768, dtype=np.uint16)
    return float(_PREFLOP[combo_index(hole)]) / 65535.0


# ---------------------------------------------------------------------------
# Postflop Monte Carlo equity (vs a uniformly random opponent hand)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(12345)
_ALL = np.arange(36)


def postflop_equity_mc(hole, board, n: int = 400, rng=None) -> float:
    """Equity (win + tie/2) of `hole` vs one random hand on a 3/4/5-card board,
    by sampling n (opponent hand, board completion) pairs without replacement."""
    rng = rng or _RNG
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    used = np.zeros(36, dtype=bool)
    used[hole] = True
    used[board] = True
    unseen = _ALL[~used]
    need = 5 - len(board)
    # random 5 + need cards per sample: argsort of random keys = a shuffle per row
    idx = np.argsort(rng.random((n, len(unseen))), axis=1)[:, : 5 + need]
    draw = unseen[idx]
    opp = draw[:, :5]
    full = np.concatenate([np.broadcast_to(board, (n, len(board))), draw[:, 5:]], axis=1)
    mine = omaha_scores(np.broadcast_to(hole, (n, 5)), full)
    theirs = omaha_scores(opp, full)
    return float(((mine > theirs).sum() + 0.5 * (mine == theirs).sum()) / n)


def warm() -> None:
    """Load every table now so the first decision pays no setup cost."""
    score_arrays()
    preflop_equity((0, 1, 2, 3, 4))
    postflop_equity_mc((0, 1, 2, 3, 4), (10, 11, 12), 50)
