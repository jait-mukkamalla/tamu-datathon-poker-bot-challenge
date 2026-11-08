"""
nit.py - local test opponent: NitBot (tight-passive).

Plays only strong hands (preflop table equity / postflop Monte Carlo equity vs a
random hand), mostly calls, rarely raises, folds a lot to bets. Deterministic given
Python's `random` seed. Runs as a server on port 5009 like the baselines.
"""

import os
import sys
import traceback

# This file lives in tools/opponents/, so look two folders up for game.py
# (tools/sim.py also puts the repo root on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from game import Action, AgentConfig, HandResult, Observation, Street

PARTICIPANT = "Local Test Opponents"
AGENT_NAME = "NitBot"


# ===== YOUR CODE HERE =====

# --- shared strength helpers (identical copy in every tools/opponents bot) ---------
# Self-contained on purpose: a single-file bot gets no helper modules from sim.py.
# Only game.py, evaluator.py and data/preflop_equity.npy (next to game.py) are used.
import random
from itertools import combinations
from math import comb

import numpy as np

import evaluator
import game as _game

_DATA = os.path.join(os.path.dirname(os.path.abspath(_game.__file__)), "data", "preflop_equity.npy")
_COMB = [[comb(n, k) for k in range(6)] for n in range(36)]
_RK = np.array([5 ** (c >> 2) for c in range(36)], dtype=np.int32)
_SK = np.array([1 << (3 * (c & 3)) for c in range(36)], dtype=np.int32)
_P2 = np.array(list(combinations(range(5), 2)), dtype=np.int64)
_T3 = np.array(list(combinations(range(5), 3)), dtype=np.int64)
_FL = np.zeros(4096, dtype=bool)
for _s in range(4):
    _FL[5 << (3 * _s)] = True
_MC_N = 120   # Monte Carlo samples per street (cached per street)


def _arrays():
    """(non-flush, flush) score arrays keyed by summed rank keys, shared by every
    opponent bot in the process (cached on the evaluator module)."""
    arr = getattr(evaluator, "_opp_score_arrays", None)
    if arr is None:
        table = evaluator.get_table()
        nf = np.zeros(5 ** 9, dtype=np.int32)
        fl = np.zeros(5 ** 9, dtype=np.int32)
        rk = [5 ** (c >> 2) for c in range(36)]
        for a, b, c, d, e in combinations(range(36), 5):
            score = table[(1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e)]
            key = rk[a] + rk[b] + rk[c] + rk[d] + rk[e]
            if (a & 3) == (b & 3) == (c & 3) == (d & 3) == (e & 3):
                fl[key] = score
            else:
                nf[key] = score
        arr = (nf, fl)
        evaluator._opp_score_arrays = arr
    return arr


def _preflop_table():
    t = getattr(evaluator, "_opp_preflop", None)
    if t is None:
        try:
            t = np.load(_DATA).astype(np.float32) / 65535.0
        except Exception:
            t = np.full(376992, 0.5, dtype=np.float32)
        evaluator._opp_preflop = t
    return t


def _scores(hole: np.ndarray, board: np.ndarray) -> np.ndarray:
    nf, fl = _arrays()
    hr, hs, br, bs = _RK[hole], _SK[hole], _RK[board], _SK[board]
    hr2 = hr[:, _P2[:, 0]] + hr[:, _P2[:, 1]]
    hs2 = hs[:, _P2[:, 0]] + hs[:, _P2[:, 1]]
    br3 = br[:, _T3[:, 0]] + br[:, _T3[:, 1]] + br[:, _T3[:, 2]]
    bs3 = bs[:, _T3[:, 0]] + bs[:, _T3[:, 1]] + bs[:, _T3[:, 2]]
    rk = hr2[:, :, None] + br3[:, None, :]
    sk = hs2[:, :, None] + bs3[:, None, :]
    return np.where(_FL[sk], fl[rk], nf[rk]).reshape(len(hole), 100).max(axis=1)


def preflop_eq(hole) -> float:
    s = sorted(hole)
    i = _COMB[s[0]][1] + _COMB[s[1]][2] + _COMB[s[2]][3] + _COMB[s[3]][4] + _COMB[s[4]][5]
    return float(_preflop_table()[i])


def equity(hole, board, rng: random.Random, n: int = _MC_N) -> float:
    """Equity (win + tie/2) vs one random hand, Monte Carlo over n samples."""
    if not board:
        return preflop_eq(hole)
    used = set(hole) | set(board)
    rem = np.array([c for c in range(36) if c not in used], dtype=np.int64)
    need = 5 - len(board)
    nrng = np.random.default_rng(rng.getrandbits(32))
    perm = rem[np.argsort(nrng.random((n, len(rem))), axis=1)[:, :5 + need]]
    opp = perm[:, :5]
    full = np.concatenate([np.broadcast_to(np.array(board, dtype=np.int64), (n, len(board))),
                           perm[:, 5:]], axis=1)
    mine = _scores(np.broadcast_to(np.array(hole, dtype=np.int64), (n, 5)), full)
    theirs = _scores(opp, full)
    return float(((mine > theirs).sum() + 0.5 * (mine == theirs).sum()) / n)


def made_category(hole, board) -> int:
    """Category (evaluator.HIGH_CARD..STRAIGHT_FLUSH) of the best made hand now,
    exactly 2 hole + 3 board cards."""
    table = evaluator.get_table()
    best = 0
    for p in combinations(hole, 2):
        for t in combinations(board, 3):
            m = (1 << p[0]) | (1 << p[1]) | (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            s = table[m]
            if s > best:
                best = s
    return evaluator.category_of(best)


def pairs_board(hole, board) -> bool:
    """A hole card pairs a board card (or a pocket pair beats/joins the board)."""
    br = {c >> 2 for c in board}
    return any((c >> 2) in br for c in hole)


def street_eq(obs, memory) -> float:
    """Equity vs random for this street, cached so each street costs one MC run."""
    key = (obs.hand_index, int(obs.street))
    cache = memory.setdefault("_eq", {})
    if key not in cache:
        if len(cache) > 8:
            cache.clear()
        cache[key] = equity(obs.hole_cards, obs.board, memory["rng"])
    return cache[key]


def pick(obs, *prefs):
    """First legal action among prefs, else check/call, else fold."""
    for a in prefs:
        if a in obs.legal_actions:
            return a
    if Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    return obs.legal_actions[0]


def init_rng(memory) -> None:
    """Per-half generator, seeded from Python's `random` (sim reseeds it each half)."""
    memory["rng"] = random.Random(random.getrandbits(32))


_arrays()
_preflop_table()

# --- policy ----------------------------------------------------------------------

RH, RP = Action.RAISE_HALF, Action.RAISE_POT


def new_half(config: AgentConfig, memory: dict) -> None:
    init_rng(memory)


def choose_action(obs: Observation, memory: dict) -> Action:
    r = memory["rng"]
    e = street_eq(obs, memory)
    nr = obs.raises_this_street
    if obs.street == Street.PREFLOP:
        if e >= 0.60 and nr < 2 and r.random() < 0.6:     # top ~5%: often raise
            return pick(obs, RH)
        need = 0.53 + 0.02 * nr                            # ~top 40% to limp/call one raise
        if obs.to_call == 0 or e >= need:
            return pick(obs, Action.CHECK_CALL)
        return pick(obs, Action.FOLD)
    if obs.to_call == 0:
        if e >= 0.80 and r.random() < 0.6:
            return pick(obs, RH)
        return pick(obs, Action.CHECK_CALL)
    if e >= 0.90 and nr < 2 and r.random() < 0.4:
        return pick(obs, RH)
    need = 0.60 + 0.04 * nr + (0.03 if obs.street == Street.RIVER else 0.0)
    if e >= need:
        return pick(obs, Action.CHECK_CALL)
    return pick(obs, Action.FOLD)


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
# Keep your per-half state in `memory`: other global variables are NOT reset
# between halves.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5009 for your own testing convenience
    port = int(os.environ.get("PORT", "5009"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
