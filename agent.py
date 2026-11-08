"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import time
import traceback

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from eqlib import equity, preflop_equity, range_equity, warm  # noqa: E402
import oppmodel
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()
warm()

# rename as your team/bot's name
PARTICIPANT = "jram"
AGENT_NAME = "joker"


# ===== YOUR CODE HERE =====

def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands).

    `memory` is a fresh, empty dict every half. Store anything you want to
    remember between hands in it, e.g. memory["opp_raises"] = 0.
    `config` holds the match settings (starting_stack, big_blind, ...).

    In the tournament, each half runs in a brand-new copy of your server,
    so nothing you learn in one half carries over to the other half.
    """
    memory["hands_played"] = 0
    memory["opp"] = oppmodel.new_state()


# Policy parameters. Thresholds are on equity vs a random hand (after shading when
# facing a bet). Offsets are ADDED to the bet/raise thresholds, so a negative offset
# means "more aggressive in this situation". Values from tools/tune.py (coordinate
# search vs a pool of our earlier bots and baselines), last retuned with range-weighted
# equity and bluffing on, and validated on held-out seeds against each opponent.
P = {
    "open_pot": 0.70, "open_half": 0.50,          # postflop, bet when checked to
    "raise_pot": 0.80, "raise_half": 0.68,        # postflop, raise when facing a bet
    "pre_open_pot": 0.62, "pre_open_half": 0.51,  # preflop, nothing to call beyond the blind
    "pre_raise_pot": 0.66, "pre_raise_half": 0.62,  # preflop, facing a raise
    "call_margin": 0.06,                          # extra equity over pot odds to call
    "shade_scale": 0.85, "shade_per_raise": 0.06,  # facing a bet: their range beats random
    "off_flop": -0.03, "off_turn": 0.0, "off_river": 0.0,  # postflop offsets by street
    "off_ip": 0.0,         # postflop offset when we act last (we are the button)
    "pre_off_btn": 0.0,    # preflop offset when we are the button
    "spr_coef": -0.03,     # offset -= spr_coef * (3 - min(SPR, 6)) / 3: >0 pushes harder when shallow
    "call_off_river": 0.02,  # extra river call margin
    "jitter": 0.0,         # std of random noise added to equity before deciding (mixing)
    # Range-weighted equity: each opponent bet/raise keeps the top q = (their raise
    # frequency) ** range_pow of hands, each call of our raise the top (1 - their
    # fold-to-raise rate) ** range_pow; the power is scaled down when showdowns show
    # their bets are not stronger than random. 0 = off (equity vs a random hand).
    "range_on": 1.0, "range_pow": 1.0,
    # Bluffing when checked to: bet when the one-step EV of betting, using the
    # opponent's fold-to-raise rate and our equity vs their calling range, beats
    # checking by more than bluff_margin * pot. 0 = off.
    "bluff_on": 1.0, "bluff_margin": 0.02,
}
try:  # LOCAL TUNING ONLY: the tournament never sets this variable
    import json as _json
    P.update(_json.loads(os.environ.get("BOT_PARAMS", "{}")))
except Exception:
    pass
# Time pacing. The judge charges the wall time of every request minus 1 ms against an
# 8 s bank per half. We measure our own functions (deciding, on_hand_end), add an
# allowance per request for the server's JSON work, and spread TIME_TARGET over the half. The rest of the bank is margin for a slow or loaded
# tournament host, where network/CPU-wait time we cannot see is also charged.
TIME_TARGET = 4.5      # seconds of estimated request time we aim to use per half
REQUEST_OVERHEAD = 3e-4  # seconds per request of server work outside our functions
MC_MAX_TIME = 0.04     # hard ceiling on Monte Carlo time for one decision (limit is 1 s)
CAP_FACTOR = 3.0       # sample cap = this x the average affordable count; early stopping
                       # means most decisions use far fewer, and pacing absorbs the rest
MIN_SAMPLES = 48
MAX_SAMPLES = 4000
CALL_OVERHEAD = 1.5e-4  # fixed seconds per MC call, independent of sample count
Z_STOP = 3.0           # stop sampling once equity is this many std errors from every threshold
# LOCAL TUNING ONLY: a fixed sample cap instead of time pacing, so parallel simulations
# (where the machine is loaded) still compare candidates at the same accuracy.
FIXED_CAP = int(os.environ.get("BOT_FIXED_CAP", "0") or 0)
HANDS_PER_HALF = 1000


def _pick(obs: Observation, *prefs: Action) -> Action:
    """First preferred action that is legal, else a safe legal fallback."""
    for a in prefs:
        if a in obs.legal_actions:
            return a
    if Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    return obs.legal_actions[0]


def _spent(memory: dict) -> float:
    """Estimated seconds charged so far this half: time measured inside our own
    functions (choose_action, on_hand_end) plus REQUEST_OVERHEAD per request for the
    server work around them (JSON parsing and replies) that they cannot see."""
    return memory.get("self_time", 0.0) + memory.get("reqs", 0) * REQUEST_OVERHEAD


def _sample_cap(obs: Observation, memory: dict) -> int:
    """Monte Carlo sample cap for this decision.

    What is left of TIME_TARGET, minus the fixed cost of the requests still to come
    (JSON, preflop, on_hand_end: measured as total time minus Monte Carlo time), is
    shared over the Monte Carlo decisions still expected this half. Early stopping
    usually ends well below the cap, and any over- or under-spend feeds back into the
    next decisions' share."""
    hands_done = max(obs.hand_index, 1)
    reqs, mcs = memory.get("reqs", 0), memory.get("mcs", 0)
    # per-hand rates, blended with priors (3.8 requests, 2 MC decisions) early on
    w = min(1.0, hands_done / 50)
    req_rate = w * reqs / hands_done + (1 - w) * 3.8
    mc_rate = max(w * mcs / hands_done + (1 - w) * 2.0, 0.5)
    spent = _spent(memory)
    fixed = (spent - memory.get("mc_time", 0.0)) / reqs if reqs else 3e-4
    left = HANDS_PER_HALF - obs.hand_index
    mc_left = TIME_TARGET - spent - left * req_rate * max(fixed, 0.0)
    per_call = mc_left / max(left * mc_rate, 1.0)
    budget = min(MC_MAX_TIME, CAP_FACTOR * per_call)
    cost = memory.get("cost", {}).get(len(obs.board), 2e-6)
    n = int((budget - CALL_OVERHEAD) / cost) if budget > CALL_OVERHEAD else MIN_SAMPLES
    return max(MIN_SAMPLES, min(MAX_SAMPLES, n))


def _opp_adjust(obs: Observation, memory: dict, pre: bool) -> tuple[float, float, float]:
    """Update the opponent model from the history and return its (clamped) shifts.
    Any failure just means no adjustment."""
    try:
        st = memory["opp"]
        oppmodel.observe(obs, st)
        return oppmodel.adjustments(st, pre)
    except Exception:
        traceback.print_exc()
        return 0.0, 0.0, 0.0


def _situation_offset(obs: Observation, pre: bool) -> float:
    """Threshold offset for street, position and stack-to-pot ratio (negative = bolder)."""
    if pre:
        off = P["pre_off_btn"] if obs.is_button else 0.0
    else:
        off = {Street.FLOP: P["off_flop"], Street.TURN: P["off_turn"],
               Street.RIVER: P["off_river"]}.get(obs.street, 0.0)
        if obs.is_button:
            off += P["off_ip"]
    spr = min(obs.my_stack, obs.opp_stack) / max(obs.pot, 1)
    return off - P["spr_coef"] * (3.0 - min(spr, 6.0)) / 3.0


def _policy(obs: Observation, memory: dict, pre: bool):
    """Build the decision rule for this spot before any equity is computed.

    Returns (act, thresholds): act(e) maps our raw equity vs a random hand to an
    action, and thresholds are the raw-equity values where that action changes.
    Knowing them up front lets the Monte Carlo stop once the decision is settled."""
    shade_adj, open_adj, margin_adj = _opp_adjust(obs, memory, pre)
    off = _situation_offset(obs, pre)
    if obs.to_call == 0:
        hi, lo = (P["pre_open_pot"], P["pre_open_half"]) if pre else (P["open_pot"], P["open_half"])
        hi, lo = hi + open_adj + off, lo + open_adj + off

        def act(e: float) -> Action:
            if e >= hi:
                return _pick(obs, Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL)
            if e >= lo:
                return _pick(obs, Action.RAISE_HALF, Action.CHECK_CALL)
            return _pick(obs, Action.CHECK_CALL)
        cuts = [hi, lo]
    else:
        # Facing a bet the opponent's range is stronger than random: shade equity down.
        # e_eff = 0.5 + (e - 0.5) * scale - sub, a rising line, so its cutoffs map
        # back to raw-equity cutoffs exactly.
        scale = P["shade_scale"]
        sub = P["shade_per_raise"] * obs.raises_this_street + shade_adj
        pot_odds = obs.to_call / (obs.pot + obs.to_call)
        hi, lo = (P["pre_raise_pot"], P["pre_raise_half"]) if pre else (P["raise_pot"], P["raise_half"])
        hi, lo = hi + off, lo + off
        margin = (P["call_margin"] + margin_adj
                  + (P["call_off_river"] if obs.street == Street.RIVER else 0.0))
        call_at = pot_odds + margin

        def act(e: float) -> Action:
            e_eff = 0.5 + (e - 0.5) * scale - sub
            if e_eff >= hi:
                return _pick(obs, Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL)
            if e_eff >= lo:
                return _pick(obs, Action.RAISE_HALF, Action.CHECK_CALL)
            if e_eff > call_at:
                return _pick(obs, Action.CHECK_CALL)
            return _pick(obs, Action.FOLD, Action.CHECK_CALL)
        cuts = [(t - 0.5 + sub) / scale + 0.5 for t in (hi, lo, call_at)] if scale > 0 else []
    # keep only cutoffs where the chosen (legal) action really changes
    eps = 1e-9
    return act, [t for t in cuts if act(t - eps) != act(t + eps)]


_BOARD_LEN = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)
Q_MIN = 0.12   # never narrow a range below this fraction per action


def _ranges(obs: Observation, memory: dict) -> list[tuple[int, float]]:
    """Range narrowings (board_len, q) implied by the opponent's actions this hand."""
    st = memory["opp"]
    f = {g: oppmodel.freqs(st, g) for g in (True, False)}
    # narrow only as far as showdowns show their actions mean strength: a bot that
    # bets random hands gets (almost) no narrowing
    pw = P["range_pow"] * oppmodel.bet_signal(st)
    if pw <= 0:
        return []
    out = []
    hist = obs.action_history
    for i, a in enumerate(hist):
        if a.is_me:
            continue
        agg, ftr = f[a.street == Street.PREFLOP]
        prev = hist[i - 1] if i > 0 else None
        if a.action in _RAISES:
            q = agg
        elif a.action == Action.CHECK_CALL and prev is not None and prev.is_me and prev.action in _RAISES:
            q = 1.0 - ftr
        else:
            continue
        out.append((_BOARD_LEN[a.street], max(Q_MIN, q ** pw)))
    return out


def _bluff(obs: Observation, memory: dict, e: float, e_call: float, f: float) -> Action | None:
    """Best bluff/semi-bluff bet when checking was the plan, or None.

    One-step EV relative to checking, in chips:
        bet B into pot P:  f * P + (1 - f) * (e_call * (P + 2B) - B)
        check:             e * P
    (ignores later streets and opponent raises; bluff_margin absorbs the rest)."""
    best, best_gain = None, P["bluff_margin"] * obs.pot
    for a in _RAISES:
        if a not in obs.legal_actions:
            continue
        b = obs.raise_to[a] - obs.my_bet
        gain = f * obs.pot + (1 - f) * (e_call * (obs.pot + 2 * b) - b) - e * obs.pot
        if gain > best_gain:
            best, best_gain = a, gain
    return best


def _rng(obs: Observation) -> np.random.Generator:
    """Random stream fixed by the spot (deal, street, position in the hand): the same
    spot gets the same samples, which makes paired comparisons of bot versions tight."""
    key = (obs.hand_index, int(obs.street), len(obs.action_history),
           sum(obs.hole_cards) * 37 + sum(obs.board))
    return np.random.default_rng(key)


def _decide(obs: Observation, memory: dict) -> Action:
    pre = obs.street == Street.PREFLOP
    act, cuts = _policy(obs, memory, pre)
    rng = _rng(obs)
    jit = P["jitter"]
    if pre:
        e = preflop_equity(obs.hole_cards)
        return act(e + rng.normal(0.0, jit) if jit > 0 else e)
    if not cuts:  # every equity gives the same action (e.g. only one legal move)
        return act(0.5)
    cap = FIXED_CAP or _sample_cap(obs, memory)
    ranges = _ranges(obs, memory) if P["range_on"] else []
    bluff = P["bluff_on"] and obs.to_call == 0 and any(a in obs.legal_actions for a in _RAISES)
    f = oppmodel.freqs(memory["opp"], False)[1] if bluff else 0.0
    extra = [(len(obs.board), max(Q_MIN, 1.0 - f))] if bluff else []
    t0 = time.perf_counter()
    e, n, ex = range_equity(obs.hole_cards, obs.board, ranges, cuts, cap, Z_STOP, rng=rng,
                            extra=extra)
    dt = time.perf_counter() - t0
    memory["mc_time"] = memory.get("mc_time", 0.0) + dt
    memory["mcs"] = memory.get("mcs", 0) + 1
    # EMA of seconds per sample by board size (river is far cheaper); tracks host speed
    costs = memory.setdefault("cost", {})
    b = max(dt - CALL_OVERHEAD, 1e-7) / n
    costs[len(obs.board)] = 0.8 * costs.get(len(obs.board), b) + 0.2 * b
    if jit > 0:
        e = e + rng.normal(0.0, jit)
    action = act(e)
    if bluff and action == Action.CHECK_CALL:
        action = _bluff(obs, memory, e, ex[0], f) or action
    return action


def choose_action(obs: Observation, memory: dict) -> Action:
    """Equity bot: preflop table + Monte Carlo postflop equity, pot-odds calls.
    Never raises: falls back to a legal action."""
    t0 = time.perf_counter()
    memory["reqs"] = memory.get("reqs", 0) + 1
    try:
        action = _decide(obs, memory)
        try:
            oppmodel.note_my_action(obs, action, memory["opp"])
        except Exception:
            traceback.print_exc()
        return action
    except Exception:
        traceback.print_exc()
        if obs.to_call == 0 and Action.CHECK_CALL in obs.legal_actions:
            return Action.CHECK_CALL
        return _pick(obs, Action.FOLD, Action.CHECK_CALL)
    finally:
        memory["self_time"] = memory.get("self_time", 0.0) + time.perf_counter() - t0


def _showdown_equity(opp_hole, board) -> float:
    return equity(opp_hole, board, (), 400)[0]


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Optional: learn from what happened.

    To save time, the judge sends finished hands along with your next
    decision request, so this always runs BEFORE choose_action() sees the
    next hand. (The last hands of a half arrive at the end of the half.)

    result has: hand_index, my_net (chips won or lost), pot,
    went_to_showdown, board (cards revealed), opp_hole (opponent's cards,
    only after a showdown, else None), winner ('me', 'opp' or 'split').
    """
    t0 = time.perf_counter()
    memory["hands_played"] = memory.get("hands_played", 0) + 1
    try:
        oppmodel.finish(result, memory["opp"], _showdown_equity)
    except Exception:
        traceback.print_exc()
    memory["self_time"] = memory.get("self_time", 0.0) + time.perf_counter() - t0

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
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

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
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
