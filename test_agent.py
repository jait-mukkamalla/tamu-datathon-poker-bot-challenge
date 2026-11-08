"""
Tests for YOUR bot (agent.py). Run them from the bundle folder:

    python -m pytest -q

They talk to your bot the same way the judge does (through its web routes,
but without starting a real server), so a passing run means the judge can
play with your bot. Add your own tests at the bottom!
"""

import importlib.util
import os
import random
import sys

import pytest
from fastapi.testclient import TestClient

# Works whether this file sits next to agent.py or inside a tests/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = HERE if os.path.isfile(os.path.join(HERE, "game.py")) else os.path.dirname(HERE)
sys.path.insert(0, BUNDLE)  # so agent.py can import game.py and evaluator.py

from game import Action, AgentConfig, Game, make_deal  # noqa: E402
from judge_engine import coerce_action  # noqa: E402  (reads answers exactly like the judge)

TOKEN = "test-token"
AUTH = {"X-Agent-Token": TOKEN}


def load_agent():
    """Load a fresh copy of agent.py (its own MEMORY and globals)."""
    spec = importlib.util.spec_from_file_location("agent_under_test", os.path.join(BUNDLE, "agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bot():
    """(agent module, test client) with the tournament's token check switched on."""
    module = load_agent()
    module.TOKEN = TOKEN
    return module, TestClient(module.app)


def start_half(client, half=1, seat=0):
    """POST /new-half the way the judge does."""
    body = {"half": half, "seat": seat, "config": AgentConfig().to_dict()}
    return client.post("/new-half", json=body, headers=AUTH)


def play_hands(client, num_hands, seed="test", raise_rate=0.3):
    """Play hands where YOUR bot is seat 0 and a random opponent is seat 1.

    Every decision goes through /act exactly like in a real match, and
    finished hands are delivered with the next /act (like the judge does).
    Returns the list of every action your bot chose.
    """
    rng = random.Random(seed)
    pending = []      # finished hands waiting to be delivered
    chosen = []
    for hand in range(num_hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": pending},
                                headers=AUTH)
                pending = []
                assert r.status_code == 200, f"/act failed: {r.json()}"
                reply = r.json()
                assert reply.get("hand_end_errors", []) == [], \
                    f"on_hand_end crashed: {reply['hand_end_errors'][0]['traceback']}"
                action = coerce_action(reply["action"])
                assert action in obs.legal_actions, \
                    f"choose_action returned {reply['action']!r}; legal were {[a.name for a in obs.legal_actions]}"
                chosen.append(action)
            else:
                # Random opponent that raises sometimes, so your bot faces bets.
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < raise_rate else Action.CHECK_CALL
            game.apply(action)
        pending.append(game.hand_result(0).to_dict())
    r = client.post("/hand-end", json={"results": pending}, headers=AUTH)  # end of half
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []
    return chosen


# ---------------------------------------------------------------------------
# Your bot plays correctly
# ---------------------------------------------------------------------------

def test_names_are_set(bot):
    _, client = bot
    info = client.get("/").json()
    assert isinstance(info["participant"], str) and info["participant"]
    assert isinstance(info["agent_name"], str) and info["agent_name"]


def test_new_half_works(bot):
    _, client = bot
    r = start_half(client)
    assert r.status_code == 200, f"new_half crashed: {r.json()}"


def test_every_decision_is_legal(bot):
    # 200 hands with a random opponent: every choose_action answer must be legal.
    _, client = bot
    start_half(client)
    play_hands(client, 200)


def test_works_in_both_seats_and_both_halves(bot):
    _, client = bot
    for half, seat in ((1, 0), (2, 1)):
        assert start_half(client, half, seat).status_code == 200
        play_hands(client, 30, seed=f"half{half}")


def test_handles_aggressive_opponent(bot):
    # An opponent that raises whenever it can: your bot must cope with
    # big bets and all-ins.
    _, client = bot
    start_half(client)
    play_hands(client, 100, seed="aggressive", raise_rate=1.0)


def test_memory_is_fresh_each_half(bot):
    module, client = bot
    start_half(client)
    play_hands(client, 10)
    start_half(client, half=2, seat=1)
    after_reset = dict(module.MEMORY)
    fresh = load_agent()
    fresh.new_half(AgentConfig(), fresh.MEMORY)
    assert after_reset.keys() == fresh.MEMORY.keys(), "memory should only hold what new_half() puts there"


# ---------------------------------------------------------------------------
# The server plumbing still works (only fails if you edited it)
# ---------------------------------------------------------------------------

def test_token_is_checked(bot):
    _, client = bot
    body = {"half": 1, "seat": 0, "config": {}}
    assert client.post("/new-half", json=body).status_code == 401
    assert client.post("/new-half", json=body, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_bad_token_is_refused_before_the_body_is_read(bot):
    # Without the right token the bot must answer 401 without parsing the body,
    # so junk sent by anyone else can't slow your bot down.
    _, client = bot
    junk = b"this is not JSON" * 1000
    assert client.post("/act", content=junk).status_code == 401
    assert client.post("/act", content=junk, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_malformed_body_is_rejected(bot):
    _, client = bot
    start_half(client)
    assert client.post("/act", content=b"not json", headers=AUTH).status_code == 400
    assert client.post("/act", json=[1, 2, 3], headers=AUTH).status_code == 400


def test_act_before_new_half_is_refused(bot):
    _, client = bot
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    assert client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH).status_code == 409


def test_crashes_are_reported_to_the_judge(bot):
    # If choose_action raises, the judge must get a 500 with your traceback.
    module, client = bot

    def broken(obs, memory):
        raise ValueError("oops")

    module.choose_action = broken
    start_half(client)
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    r = client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH)
    assert r.status_code == 500
    assert "ValueError: oops" in r.json()["error"] and "Traceback" in r.json()["traceback"]


def test_reads_port_from_environment():
    source = open(os.path.join(BUNDLE, "agent.py"), encoding="utf-8").read()
    assert 'os.environ.get("PORT"' in source, "agent.py must read its port from $PORT"
    assert '0.0.0.0' in source, "agent.py must listen on 0.0.0.0 so the judge can reach it"


# ---------------------------------------------------------------------------
# Add your own tests below, e.g. "with two aces preflop my bot raises".
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# eqlib / preflop table
# ---------------------------------------------------------------------------

def test_vectorized_scores_match_evaluator():
    import numpy as np
    from eqlib import omaha_scores
    from evaluator import omaha_score

    rng = np.random.default_rng(1)
    d = np.argsort(rng.random((300, 36)), axis=1)[:, :10]
    got = omaha_scores(d[:, :5], d[:, 5:])
    exp = [omaha_score(list(r[:5]), list(r[5:])) for r in d]
    assert list(got) == exp


def test_preflop_equity_table():
    from eqlib import preflop_equity
    from evaluator import cards_from_str

    good = preflop_equity(cards_from_str("AsAhKsKhQs"))
    bad = preflop_equity(cards_from_str("6c7d8h9sTc"))
    assert 0.6 < good < 0.7 and good > bad
    # suit-isomorphic hands share one value
    assert preflop_equity(cards_from_str("AcAdKcKdQc")) == good


def test_postflop_equity_mc_sane():
    from eqlib import postflop_equity_mc
    # board 3-3-3-5-6: holding the fourth 3 (quads) must beat a hand with no 3
    board = (12, 13, 14, 20, 24)
    strong = postflop_equity_mc((0, 5, 10, 15, 21), board, 300)
    weak = postflop_equity_mc((0, 1, 4, 5, 8), board, 300)
    assert 0.0 <= weak <= 1.0 and 0.0 <= strong <= 1.0
    assert strong > weak


# ---------------------------------------------------------------------------
# opponent model
# ---------------------------------------------------------------------------

def _hist(*items):
    from game import ObservedAction, Street
    return tuple(ObservedAction(Street[s], me, Action[a], 0, 0) for s, me, a in items)


class _Obs:
    def __init__(self, hand_index, street, history):
        from game import Street
        self.hand_index, self.street, self.action_history = hand_index, Street[street], history


class _Res:
    def __init__(self, hand_index, winner="me", showdown=False, opp_hole=None, board=()):
        self.hand_index, self.winner, self.went_to_showdown = hand_index, winner, showdown
        self.opp_hole, self.board = opp_hole, board


def test_opp_model_counts_each_action_once():
    import oppmodel
    st = oppmodel.new_state()
    h1 = _hist(("PREFLOP", True, "RAISE_POT"), ("PREFLOP", False, "CHECK_CALL"))
    oppmodel.observe(_Obs(0, "FLOP", h1), st)
    oppmodel.observe(_Obs(0, "FLOP", h1), st)   # same history again: no double count
    assert st["agg"]["pre"] == [0, 1] and st["ftr"]["pre"] == [0, 1]
    h2 = h1 + _hist(("FLOP", False, "RAISE_HALF"))
    oppmodel.observe(_Obs(0, "FLOP", h2), st)
    assert st["agg"]["post"] == [1, 1] and st["opp_bet_post"]


def test_opp_model_fold_to_our_raise_is_settled_at_hand_end():
    import oppmodel
    st = oppmodel.new_state()
    oppmodel.observe(_Obs(3, "FLOP", ()), st)
    oppmodel.note_my_action(_Obs(3, "FLOP", ()), Action.RAISE_POT, st)
    oppmodel.finish(_Res(3, winner="me", showdown=False), st)
    assert st["ftr"]["post"] == [1, 1] and st["pending"] is None


def test_opp_model_showdown_strength_and_clamps():
    import oppmodel
    st = oppmodel.new_state()
    oppmodel.observe(_Obs(0, "RIVER", _hist(("RIVER", False, "RAISE_POT"))), st)
    oppmodel.finish(_Res(0, "opp", True, (0, 1, 2, 3, 4), (5, 6, 7, 8, 9)), st, lambda h, b: 0.95)
    assert st["sd_bet"] == [0.95, 1]
    # extreme counts still give clamped, finite shifts
    st["hands"] = 500
    st["agg"]["post"], st["ftr"]["post"] = [10 ** 6, 10 ** 6], [10 ** 6, 10 ** 6]
    st["sd_bet"], st["sd_call"] = [10 ** 6, 10 ** 6], [0.0, 10 ** 6]
    for pre in (True, False):
        s, o, m = oppmodel.adjustments(st, pre)
        assert abs(s) <= 0.06 and abs(o) <= 0.06 and abs(m) <= 0.02


def test_opp_model_no_shift_during_warmup():
    import oppmodel
    st = oppmodel.new_state()
    st["agg"]["post"] = [50, 50]
    assert oppmodel.adjustments(st, False) == (0.0, 0.0, 0.0)   # 0 hands seen yet


def test_bot_survives_a_long_match_with_model(bot):
    _, client = bot
    start_half(client)
    play_hands(client, 60)
    start_half(client, half=2, seat=1)
    play_hands(client, 40, raise_rate=0.8)


def test_situation_offset_zero_when_offsets_zero_and_spr_term(bot):
    from game import Street

    module, _ = bot

    class O:
        street, is_button, my_stack, opp_stack, pot = Street.RIVER, True, 10, 10, 100

    saved = dict(module.P)
    try:
        module.P.update({k: 0.0 for k in ("off_flop", "off_turn", "off_river", "off_ip",
                                          "pre_off_btn", "spr_coef")})
        assert module._situation_offset(O, False) == 0.0
        module.P.update(spr_coef=0.03, off_river=-0.02, off_ip=-0.01)
        assert abs(module._situation_offset(O, False) - (-0.03 - 0.03 * (3 - 0.1) / 3)) < 1e-9
    finally:
        module.P.clear()
        module.P.update(saved)


# ---------------------------------------------------------------------------
# fast equity: dense tables, pools, river pair table, sequential stopping
# ---------------------------------------------------------------------------

def test_dense_scores_order_like_evaluator():
    import numpy as np
    from eqlib import omaha_dense
    from evaluator import omaha_score

    rng = np.random.default_rng(2)
    d = np.argsort(rng.random((400, 36)), axis=1)[:, :15]
    a, b = omaha_dense(d[:, :5], d[:, 10:]), omaha_dense(d[:, 5:10], d[:, 10:])
    for r, x, y in zip(d, a, b):
        ex = omaha_score(list(r[:5]), list(r[10:])) - omaha_score(list(r[5:10]), list(r[10:]))
        assert np.sign(int(x) - int(y)) == np.sign(ex)


def _exact_river(hole, board):
    from itertools import combinations
    from evaluator import omaha_score
    used = set(hole) | set(board)
    unseen = [c for c in range(36) if c not in used]
    mine = omaha_score(list(hole), list(board))
    w = t = n = 0
    for opp in combinations(unseen, 5):
        s = omaha_score(list(opp), list(board))
        w += mine > s
        t += mine == s
        n += 1
    return (w + 0.5 * t) / n


def test_river_equity_matches_exact():
    from eqlib import equity
    for hole, board in (((0, 5, 10, 15, 21), (12, 13, 14, 20, 24)),
                        ((1, 6, 11, 16, 30), (2, 7, 22, 27, 35))):
        import numpy as np
        got = equity(hole, board, (), 4000, rng=np.random.default_rng(7))[0]
        assert abs(got - _exact_river(hole, board)) < 0.03


def test_flop_turn_equity_reasonable():
    from eqlib import equity
    # suit-isomorphic boards give the same equity up to noise
    import numpy as np
    a = equity((32, 33, 28, 29, 24), (0, 5, 10), (), 4000, rng=np.random.default_rng(8))[0]
    b = equity((33, 32, 29, 28, 25), (1, 4, 11), (), 4000, rng=np.random.default_rng(9))[0]
    assert 0.0 < a < 1.0 and abs(a - b) < 0.05


def test_sequential_stopping_saves_samples_only_when_safe():
    import numpy as np
    from eqlib import FIRST_CHUNK, equity
    board = (12, 13, 14, 20, 24)
    e, n = equity((0, 5, 10, 15, 21), board, (0.4,), 2000, rng=np.random.default_rng(3))
    assert n == FIRST_CHUNK and e > 0.9   # quads vs a 0.4 cutoff: settled at once
    e0 = equity((0, 1, 4, 5, 8), board, (), 20000, rng=np.random.default_rng(1))[0]
    used = [equity((0, 1, 4, 5, 8), board, (e0,), 2000, rng=np.random.default_rng(s))[1]
            for s in range(10)]                              # cutoff right at the equity
    assert sorted(used)[5] > 500   # nearly always runs long (an early stop is a rare fluke)


def _old_rule(module, obs, e, shade_adj, open_adj, margin_adj, pre):
    """The pre-refactor decision rule, to check _policy decides exactly the same."""
    P, pick = module.P, module._pick
    off = module._situation_offset(obs, pre)
    if obs.to_call == 0:
        hi, lo = (P["pre_open_pot"], P["pre_open_half"]) if pre else (P["open_pot"], P["open_half"])
        hi, lo = hi + open_adj + off, lo + open_adj + off
        if e >= hi:
            return pick(obs, Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL)
        if e >= lo:
            return pick(obs, Action.RAISE_HALF, Action.CHECK_CALL)
        return pick(obs, Action.CHECK_CALL)
    e_eff = 0.5 + (e - 0.5) * P["shade_scale"] - P["shade_per_raise"] * obs.raises_this_street - shade_adj
    pot_odds = obs.to_call / (obs.pot + obs.to_call)
    hi, lo = (P["pre_raise_pot"], P["pre_raise_half"]) if pre else (P["raise_pot"], P["raise_half"])
    hi, lo = hi + off, lo + off
    margin = P["call_margin"] + margin_adj + (P["call_off_river"] if obs.street.name == "RIVER" else 0.0)
    if e_eff >= hi:
        return pick(obs, Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL)
    if e_eff >= lo:
        return pick(obs, Action.RAISE_HALF, Action.CHECK_CALL)
    if e_eff > pot_odds + margin:
        return pick(obs, Action.CHECK_CALL)
    return pick(obs, Action.FOLD, Action.CHECK_CALL)


def test_policy_matches_old_rule_and_cutoffs_are_exact(bot):
    import random as _r
    from game import Street
    module, _ = bot
    rng = _r.Random(5)
    grid = [i / 200 for i in range(201)]
    checked = 0
    for hand in range(150):
        game = Game(make_deal("pol", hand), button_seat=hand % 2, hand_index=hand)
        memory = {}
        module.new_half(AgentConfig(), memory)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            pre = obs.street == Street.PREFLOP
            adj = (rng.uniform(-.06, .06), rng.uniform(-.06, .06), rng.uniform(-.02, .02))
            module._opp_adjust = lambda o, m, p, adj=adj: adj
            act, cuts = module._policy(obs, memory, pre)
            for e in grid:
                assert act(e) == _old_rule(module, obs, e, *adj, pre)
            for t in cuts:   # every cutoff changes the action; none are missing
                assert act(t - 1e-9) != act(t + 1e-9)
            pts = sorted([0.0, 1.0] + [t for t in cuts if 0 < t < 1])
            for a, b in zip(pts, pts[1:]):
                mid = [a + (b - a) * f for f in (0.01, 0.5, 0.99)]
                assert len({act(x) for x in mid}) == 1
            checked += 1
            legal = obs.legal_actions
            game.apply(rng.choice(legal))
    assert checked > 300


def test_own_time_is_measured(bot):
    module, client = bot
    start_half(client)
    play_hands(client, 20)
    assert module.MEMORY["self_time"] > 0 and module.MEMORY["reqs"] > 0
    assert module._spent(module.MEMORY) > module.MEMORY["self_time"]


def test_server_plumbing_is_unchanged():
    import subprocess
    head = subprocess.run(["git", "show", "HEAD:agent.py"], capture_output=True, text=True,
                          encoding="utf-8", cwd=BUNDLE)
    if head.returncode != 0:
        pytest.skip("no git history here")
    mark = "# ===== SERVER PLUMBING"
    cur = open(os.path.join(BUNDLE, "agent.py"), encoding="utf-8").read()
    assert cur[cur.index(mark):] == head.stdout[head.stdout.index(mark):]

