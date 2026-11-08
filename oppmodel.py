"""
oppmodel.py - small, clamped opponent model for the poker bot.

It is fed from two places:
  observe(obs, st)        at every decision: counts the opponent's new actions
                          from obs.action_history (never double counts).
  finish(result, st, eq)  when a hand ends: settles what could not be seen in a
                          later observation (opponent folds / all-in calls to our
                          last raise) and records showdown strength.

adjustments(st, pre) turns the counts into three small shifts. Every estimate is
shrunk toward a prior and every shift is clamped, so a noisy read cannot move the
bot far. All of it is wrapped by the caller in try/except.
"""

from __future__ import annotations

from game import Action, Street

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

# prior mean, pseudo-count (how many observations the prior is "worth")
PRIOR_AGG = {"pre": (0.45, 15.0), "post": (0.30, 15.0)}   # P(raise | opp bet/raise/call/check)
PRIOR_FTR = {"pre": (0.30, 10.0), "post": (0.35, 10.0)}   # P(fold | opp faces our raise)
PRIOR_BET_EQ = (0.65, 5.0)    # mean river equity of the hands they bet/raise with at showdown
PRIOR_CALL_EQ = (0.55, 5.0)   # same, for hands that only called
WARMUP_HANDS = 30             # shifts ramp up linearly over this many hands


def new_state() -> dict:
    """Fresh model state (kept in memory["opp"])."""
    return {
        "hands": 0,
        "agg": {"pre": [0, 0], "post": [0, 0]},   # [raises, actions]
        "ftr": {"pre": [0, 0], "post": [0, 0]},   # [folds, times they faced our raise]
        "sd_bet": [0.0, 0],                        # [sum of equity, count]
        "sd_call": [0.0, 0],
        # per-hand scratch
        "hand": -1, "seen": 0, "pending": None, "opp_bet_post": False,
    }


def _grp(street: Street) -> str:
    return "pre" if street == Street.PREFLOP else "post"


def _reset_hand(st: dict, hand_index: int) -> None:
    st["hand"], st["seen"], st["pending"], st["opp_bet_post"] = hand_index, 0, None, False


def observe(obs, st: dict) -> None:
    """Count the opponent's actions that are new since the last observation."""
    if st["hand"] != obs.hand_index:
        _reset_hand(st, obs.hand_index)
    hist = obs.action_history
    for i in range(st["seen"], len(hist)):
        a = hist[i]
        if a.is_me:
            continue
        g = _grp(a.street)
        prev = hist[i - 1] if i > 0 else None
        faced_raise = prev is not None and prev.is_me and prev.action in RAISES
        if faced_raise:
            st["ftr"][g][1] += 1
            st["ftr"][g][0] += a.action == Action.FOLD
            st["pending"] = None  # their answer to our raise is now known
        if a.action != Action.FOLD:
            st["agg"][g][1] += 1
            st["agg"][g][0] += a.action in RAISES
        if a.action in RAISES and g == "post":
            st["opp_bet_post"] = True
    st["seen"] = len(hist)


def note_my_action(obs, action, st: dict) -> None:
    """Remember a raise of ours whose answer may never be observed (fold / all-in call)."""
    if st["hand"] != obs.hand_index:
        _reset_hand(st, obs.hand_index)
    st["pending"] = _grp(obs.street) if action in RAISES else None


def finish(result, st: dict, equity_fn=None) -> None:
    """Hand over. `equity_fn(opp_hole, board) -> float` rates a revealed hand."""
    st["hands"] += 1
    if st["hand"] == result.hand_index:
        g = st["pending"]
        if g is not None:
            folded = result.winner == "me" and not result.went_to_showdown
            st["ftr"][g][1] += 1
            st["ftr"][g][0] += folded
        if (result.went_to_showdown and result.opp_hole and equity_fn is not None
                and len(result.board) == 5):
            eq = equity_fn(tuple(result.opp_hole), tuple(result.board))
            slot = st["sd_bet"] if st["opp_bet_post"] else st["sd_call"]
            slot[0] += eq
            slot[1] += 1
    st["pending"] = None


def _est(hits: float, n: float, prior: tuple[float, float]) -> float:
    p, k = prior
    return (hits + p * k) / (n + k)


def _clamp(x: float, lim: float) -> float:
    return max(-lim, min(lim, x))


def adjustments(st: dict, pre: bool) -> tuple[float, float, float]:
    """Returns (shade_adj, open_adj, margin_adj), each already clamped and ramped.

    shade_adj  > 0: opponent's bets are stronger than assumed, so shade our equity
                    down more when facing a bet (fold more).
    open_adj   > 0: raise our thresholds for betting when checked to (bet less);
                    < 0 bets more (they fold a lot / call too weakly).
    margin_adj > 0: need more equity over pot odds to call.
    """
    g = "pre" if pre else "post"
    agg = _est(*st["agg"][g], PRIOR_AGG[g])
    ftr = _est(*st["ftr"][g], PRIOR_FTR[g])
    agg0, ftr0 = PRIOR_AGG[g][0], PRIOR_FTR[g][0]

    shade = -_clamp((agg - agg0) * 0.2, 0.03)          # very aggressive -> bets are weaker
    open_ = -_clamp((ftr - ftr0) * 0.3, 0.04)          # they fold a lot -> bet more
    margin = -_clamp((agg - agg0) * 0.1, 0.02)
    if not pre:
        bet_eq = _est(st["sd_bet"][0], st["sd_bet"][1], PRIOR_BET_EQ)
        call_eq = _est(st["sd_call"][0], st["sd_call"][1], PRIOR_CALL_EQ)
        shade += _clamp((bet_eq - PRIOR_BET_EQ[0]) * 0.5, 0.04)   # strong bets -> fold more
        open_ += _clamp((call_eq - PRIOR_CALL_EQ[0]) * 0.3, 0.03)  # strong calls -> bet less
    ramp = min(1.0, st["hands"] / WARMUP_HANDS)
    return (_clamp(shade, 0.06) * ramp, _clamp(open_, 0.06) * ramp, _clamp(margin, 0.02) * ramp)
