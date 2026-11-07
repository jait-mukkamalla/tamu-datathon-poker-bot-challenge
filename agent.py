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
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from eqlib import postflop_equity, preflop_equity, warm  # noqa: E402
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()
warm()

# rename as your team/bot's name
PARTICIPANT = "TotallyAwesomeDatathonParticipant"
AGENT_NAME = "JustAsAwesomeAgent"


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


# Tunable thresholds (equity vs a random hand, after shading for opponent aggression)
OPEN_POT, OPEN_HALF = 0.72, 0.58      # bet when checked to
RAISE_POT_T, RAISE_HALF_T = 0.80, 0.68  # raise when facing a bet
CALL_MARGIN = 0.05                    # extra equity needed over pot odds to call
SHADE_SCALE, SHADE_PER_RAISE = 0.85, 0.04
PRE_OPEN_POT, PRE_OPEN_HALF = 0.62, 0.55


def _pick(obs: Observation, *prefs: Action) -> Action:
    """First preferred action that is legal, else a safe legal fallback."""
    for a in prefs:
        if a in obs.legal_actions:
            return a
    if Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    return obs.legal_actions[0]


def _equity(obs: Observation) -> float:
    if obs.street == Street.PREFLOP:
        return preflop_equity(obs.hole_cards)
    return postflop_equity(obs.hole_cards, obs.board)


def _decide(obs: Observation) -> Action:
    e = _equity(obs)
    if obs.to_call == 0:
        hi, lo = (PRE_OPEN_POT, PRE_OPEN_HALF) if obs.street == Street.PREFLOP else (OPEN_POT, OPEN_HALF)
        if e >= hi:
            return _pick(obs, Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL)
        if e >= lo:
            return _pick(obs, Action.RAISE_HALF, Action.CHECK_CALL)
        return _pick(obs, Action.CHECK_CALL)

    # Facing a bet: the opponent's range is stronger than random, so shade equity down.
    e_eff = 0.5 + (e - 0.5) * SHADE_SCALE - SHADE_PER_RAISE * obs.raises_this_street
    pot_odds = obs.to_call / (obs.pot + obs.to_call)
    if obs.street == Street.PREFLOP:
        # Preflop equities live in a narrow band (0.1-0.7), so rescale the raise bars.
        hi, lo = RAISE_POT_T - 0.14, RAISE_HALF_T - 0.10
    else:
        hi, lo = RAISE_POT_T, RAISE_HALF_T
    if e_eff >= hi:
        return _pick(obs, Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL)
    if e_eff >= lo:
        return _pick(obs, Action.RAISE_HALF, Action.CHECK_CALL)
    if e_eff > pot_odds + CALL_MARGIN:
        return _pick(obs, Action.CHECK_CALL)
    return _pick(obs, Action.FOLD, Action.CHECK_CALL)


def choose_action(obs: Observation, memory: dict) -> Action:
    """Equity-bucket heuristic: preflop table + postflop made-hand table, pot-odds
    calls, value raises only (no bluffs). Never raises: falls back to a legal action."""
    try:
        return _decide(obs)
    except Exception:
        traceback.print_exc()
        if obs.to_call == 0 and Action.CHECK_CALL in obs.legal_actions:
            return Action.CHECK_CALL
        return _pick(obs, Action.FOLD, Action.CHECK_CALL)


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Optional: learn from what happened.

    To save time, the judge sends finished hands along with your next
    decision request, so this always runs BEFORE choose_action() sees the
    next hand. (The last hands of a half arrive at the end of the half.)

    result has: hand_index, my_net (chips won or lost), pot,
    went_to_showdown, board (cards revealed), opp_hole (opponent's cards,
    only after a showdown, else None), winner ('me', 'opp' or 'split').
    """
    memory["hands_played"] += 1

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
