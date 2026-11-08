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
    """Equity (win + tie/2) of `hole` vs one random hand on a 3/4/5-card board
    from n samples (fixed count, no early stopping). See equity()."""
    return equity(hole, board, (), n, rng=rng)[0]


# ---------------------------------------------------------------------------
# Fast scoring: dense perfect-hash tables
# ---------------------------------------------------------------------------
# Rank multipliers chosen so that every 5-card rank multiset (at most 4 of a
# rank) has a distinct sum, with a much smaller key range than 5**rank: keys
# fit in 24,919 slots, so both tables together are one 100 KB int16 array that
# stays in cache. Scores are dense ranks of the evaluator scores (same order,
# same ties), so they are only comparable to each other, not to omaha_score().

MULT = np.array([0, 1, 5, 22, 94, 312, 992, 2422, 5624], dtype=np.int32)
D_RK = MULT[np.arange(36) >> 2]          # rank key per card
D_SK = CARD_SK                           # suit key per card (flush <=> 5 << 3s)
SUIT = (np.arange(36) & 3).astype(np.int8)

_DENSE: np.ndarray | None = None         # int16: [non-flush keys | flush keys]
_FOFF: np.ndarray | None = None          # int32 (4096,): suit-key sum -> 0 or flush offset


def _build_dense() -> None:
    global _DENSE, _FOFF
    combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    table = get_table()
    masks = (np.int64(1) << combos).sum(axis=1)
    scores = np.fromiter((table[m] for m in masks.tolist()), dtype=np.int64, count=len(masks))
    keys = D_RK[combos].sum(axis=1)
    suits = combos & 3
    isfl = (suits == suits[:, :1]).all(axis=1)
    _, dense = np.unique(scores, return_inverse=True)
    size = int(keys.max()) + 1
    tab = np.zeros(2 * size, dtype=np.int16)
    tab[keys[~isfl]] = dense[~isfl]
    tab[size + keys[isfl]] = dense[isfl]
    foff = np.zeros(4096, dtype=np.int32)
    for s in range(4):
        foff[5 << (3 * s)] = size
    _DENSE, _FOFF = tab, foff


def dense_tables() -> tuple[np.ndarray, np.ndarray]:
    if _DENSE is None:
        _build_dense()
    return _DENSE, _FOFF


def omaha_dense(hole: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Like omaha_scores() but returns dense ranks (int16). (N,5),(N,5) -> (N,)."""
    tab, foff = dense_tables()
    hr, hs = D_RK[hole], D_SK[hole]
    br, bs = D_RK[board], D_SK[board]
    hr2 = hr[:, PAIRS[:, 0]] + hr[:, PAIRS[:, 1]]
    hs2 = hs[:, PAIRS[:, 0]] + hs[:, PAIRS[:, 1]]
    br3 = br[:, TRIPLES[:, 0]] + br[:, TRIPLES[:, 1]] + br[:, TRIPLES[:, 2]]
    bs3 = bs[:, TRIPLES[:, 0]] + bs[:, TRIPLES[:, 1]] + bs[:, TRIPLES[:, 2]]
    idx = hr2[:, :, None] + br3[:, None, :] + foff[hs2[:, :, None] + bs3[:, None, :]]
    return tab[idx].reshape(len(hole), 100).max(axis=1)


# ---------------------------------------------------------------------------
# Pre-shuffled sample pools
# ---------------------------------------------------------------------------
# Instead of shuffling the unseen cards on every call (argsort of random keys),
# each call picks random rows of a pool of partial shuffles made at startup.
# Unseen cards: 28 on the flop, 27 on the turn, 26 on the river (and at a
# showdown). A row holds positions into the sorted unseen-card list: the first 5
# are the opponent's hand, the rest complete the board.

POOL_ROWS = 200_000
_POOLS: dict[int, np.ndarray] = {}
_RIVER_PAIRS: np.ndarray | None = None   # (POOL_ROWS, 10) pair ids of the river pool's hands
_IA, _IB = np.triu_indices(26, 1)        # the 325 pairs of 26 unseen river cards


def _make_pool(u: int, k: int, seed: int) -> np.ndarray:
    """POOL_ROWS rows of the first k positions of a uniform shuffle of range(u)."""
    rng = np.random.default_rng(seed)
    a = np.tile(np.arange(u, dtype=np.int8), (POOL_ROWS, 1))
    rows = np.arange(POOL_ROWS)
    for i in range(k):  # partial Fisher-Yates, all rows at once
        j = rng.integers(i, u, POOL_ROWS)
        tmp = a[:, i].copy()
        a[:, i] = a[rows, j]
        a[rows, j] = tmp
    return np.ascontiguousarray(a[:, :k])


def _pool(u: int) -> np.ndarray:
    p = _POOLS.get(u)
    if p is None:
        p = _POOLS[u] = _make_pool(u, 5 + (u - 26), 1000 + u)  # opp hand + board cards to come
    return p


def _river_pairs() -> np.ndarray:
    global _RIVER_PAIRS
    if _RIVER_PAIRS is None:
        pos = _pool(26).astype(np.int64)
        pid = np.full((26, 26), -1, dtype=np.int64)
        pid[_IA, _IB] = np.arange(len(_IA))
        pid[_IB, _IA] = np.arange(len(_IA))
        _RIVER_PAIRS = np.stack([pid[pos[:, a], pos[:, b]] for a, b in PAIRS], axis=1).astype(np.int16)
    return _RIVER_PAIRS


# ---------------------------------------------------------------------------
# Equity with sequential stopping
# ---------------------------------------------------------------------------

FIRST_CHUNK = 48     # samples before the first look
VAR_FLOOR = 0.01     # floor on the per-sample variance used for the standard error


def _sampler(hole, board):
    """Return draw(rows) -> per-sample outcomes (1 win, .5 tie, 0 loss) for pool rows."""
    tab, foff = dense_tables()
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    used = np.zeros(36, dtype=bool)
    used[hole] = True
    used[board] = True
    unseen = _ALL[~used]
    nb = len(board)
    hr, hs = D_RK[hole], D_SK[hole]
    mr = (hr[PAIRS[:, 0]] + hr[PAIRS[:, 1]])[None, :, None]    # our 10 pairs, (1,10,1)

    if nb == 5:
        # River: the board is fixed, so score each of the 325 unseen pairs against
        # its best board triple once; an opponent hand is the max over its 10 pairs.
        br, bs = D_RK[board], D_SK[board]
        tr = br[TRIPLES[:, 0]] + br[TRIPLES[:, 1]] + br[TRIPLES[:, 2]]
        ts = bs[TRIPLES[:, 0]] + bs[TRIPLES[:, 1]] + bs[TRIPLES[:, 2]]
        a, b = unseen[_IA], unseen[_IB]
        pr, ps = D_RK[a] + D_RK[b], D_SK[a] + D_SK[b]
        S = tab[pr[:, None] + tr[None, :] + foff[ps[:, None] + ts[None, :]]].max(axis=1)
        ms = hs[PAIRS[:, 0]] + hs[PAIRS[:, 1]]
        mine = tab[mr[0] + tr[None, :] + foff[ms[:, None] + ts[None, :]]].max()
        pairs = _river_pairs()

        def draw(rows):
            opp = S[pairs[rows]].max(axis=1)
            return (opp < mine) + 0.5 * (opp == mine)
        return draw

    # Flop/turn: sample (opponent hand, board completion). A pair+triple is a flush
    # iff the pair is suited and the triple is one suit, the same one, so compare
    # small suit codes (pair: suit or 4, triple: suit or 5) instead of gathering a
    # flush flag for all 100 combinations.
    pool = _pool(len(unseen))
    size = int(_FOFF.max())
    hsu = SUIT[hole]
    mc = np.where(hsu[PAIRS[:, 0]] == hsu[PAIRS[:, 1]], hsu[PAIRS[:, 0]], 4).astype(np.int8)
    mc = mc[None, :, None]

    def draw(rows):
        n = len(rows)
        cards = unseen[pool[rows]]                                  # (n, 5 + cards to come)
        full = np.concatenate([np.broadcast_to(board, (n, nb)), cards[:, 5:]], axis=1)
        br, bsu = D_RK[full], SUIT[full]
        br3 = (br[:, TRIPLES[:, 0]] + br[:, TRIPLES[:, 1]] + br[:, TRIPLES[:, 2]])[:, None, :]
        s0, s1, s2 = bsu[:, TRIPLES[:, 0]], bsu[:, TRIPLES[:, 1]], bsu[:, TRIPLES[:, 2]]
        tc = np.where((s0 == s1) & (s1 == s2), s0, 5).astype(np.int8)[:, None, :]
        k = mr + br3
        np.add(k, size, out=k, where=(mc == tc))
        mine = tab[k].reshape(n, 100).max(axis=1)
        o = cards[:, :5]
        orr, osu = D_RK[o], SUIT[o]
        k = (orr[:, PAIRS[:, 0]] + orr[:, PAIRS[:, 1]])[:, :, None] + br3
        a, b = osu[:, PAIRS[:, 0]], osu[:, PAIRS[:, 1]]
        oc = np.where(a == b, a, 4).astype(np.int8)[:, :, None]
        np.add(k, size, out=k, where=(oc == tc))
        theirs = tab[k].reshape(n, 100).max(axis=1)
        return (mine > theirs) + 0.5 * (mine == theirs)
    return draw


def equity(hole, board, thresholds=(), cap: int = 300, z: float = 3.0,
           rng=None) -> tuple[float, int]:
    """Equity (win + tie/2) of `hole` vs one uniformly random hand on a 3/4/5-card
    board, by Monte Carlo. Returns (estimate, samples used).

    With `thresholds` (equity values where our decision changes), sampling stops
    as soon as the estimate is more than z standard errors from every threshold:
    the decision is then settled and more samples would not change it. Looks
    happen after FIRST_CHUNK samples and then each time the count doubles, up to
    `cap`. Without thresholds exactly `cap` samples are used."""
    rng = rng or _RNG
    cap = max(1, int(cap))
    draw = _sampler(hole, board)
    th = [t for t in thresholds if 0.0 < t < 1.0]
    if not th:
        x = draw(rng.integers(0, POOL_ROWS, cap))
        return float(x.mean()), cap
    s = s2 = 0.0
    n = 0
    chunk = min(FIRST_CHUNK, cap)
    while True:
        x = draw(rng.integers(0, POOL_ROWS, chunk))
        s += float(x.sum())
        s2 += float((x * x).sum())
        n += chunk
        p = s / n
        if n >= cap:
            return p, n
        se = (max(s2 / n - p * p, VAR_FLOOR) / n) ** 0.5
        if min(abs(p - t) for t in th) > z * se:
            return p, n
        chunk = min(n, cap - n)


def warm() -> None:
    """Load every table now so the first decision pays no setup cost."""
    dense_tables()
    for u in (26, 27, 28):
        _pool(u)
    _river_pairs()
    preflop_equity((0, 1, 2, 3, 4))
    for board in ((10, 11, 12), (10, 11, 12, 13), (10, 11, 12, 13, 14)):
        equity((0, 1, 2, 3, 4), board, (0.5,), 200)
