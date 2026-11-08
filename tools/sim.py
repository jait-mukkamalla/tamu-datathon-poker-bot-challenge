"""
tools/sim.py - fast in-process duplicate matches for tuning (LOCAL ONLY).

Plays the same duplicate format as judge_engine.py (same deals in both halves,
seats swapped, fresh memory each half) but calls the bots' functions directly
instead of over HTTP, and runs many matches in parallel processes.

A bot is a spec (directory, params): `directory` holds that bot's agent.py and
its own eqlib.py / oppmodel.py, `params` (dict or None) is merged into the bot's
module-level P dict if it has one.

    from tools.sim import run_matches
    res = run_matches([(spec_a, spec_b, seed, deals), ...], workers=20)
    # -> list of (mean bb/deal for spec_a, stderr)

CLI:  python tools/sim.py DIR_A DIR_B --seeds 1-10 --deals 1000 [--params-a JSON]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from game import AgentConfig, Game, button_for_deal, default_action, make_deal  # noqa: E402

BIG_BLIND = 2
_SHARED: dict[str, dict] = {}    # dir -> {"eqlib": module, "oppmodel": module}
_BOTS: dict[str, object] = {}    # spec key -> agent module
_COUNTER = [0]


def _load_file(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_bot(directory: str, params: dict | None = None):
    """Load directory/agent.py as a fresh module, wired to that directory's helpers."""
    directory = os.path.abspath(directory)
    agent_file = os.path.join(directory, "agent.py")
    if directory.endswith(".py"):  # a single-file bot such as baseline-bots/*.py
        agent_file, directory = directory, os.path.dirname(directory)
    key = agent_file + "|" + json.dumps(params or {}, sort_keys=True)
    if key in _BOTS:
        return _BOTS[key]
    if len(_BOTS) > 12:  # keep memory bounded when tuning many candidates
        _BOTS.pop(next(iter(_BOTS)))
    saved = {k: sys.modules.get(k) for k in ("eqlib", "oppmodel")}
    try:
        shared = _SHARED.setdefault(directory, {})
        for helper in ("eqlib", "oppmodel"):
            path = os.path.join(directory, helper + ".py")
            if os.path.exists(path):
                if helper not in shared:
                    _COUNTER[0] += 1
                    shared[helper] = _load_file(f"{helper}_{_COUNTER[0]}", path)
                sys.modules[helper] = shared[helper]
            else:
                sys.modules.pop(helper, None)
        _COUNTER[0] += 1
        old_env = os.environ.pop("BOT_PARAMS", None)
        mod = _load_file(f"bot_{_COUNTER[0]}", agent_file)
        if old_env is not None:
            os.environ["BOT_PARAMS"] = old_env
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    if params and hasattr(mod, "P"):
        mod.P.update(params)
    _BOTS[key] = mod
    return mod


class _Seat:
    def __init__(self, mod):
        self.mod, self.memory, self.errors = mod, {}, 0

    def new_half(self):
        self.memory = {}
        self.mod.new_half(AgentConfig(), self.memory)

    def act(self, obs):
        try:
            a = self.mod.choose_action(obs, self.memory)
        except Exception:
            a = None
            self.errors += 1
        if a not in obs.legal_actions:
            self.errors += a is not None
            return default_action(obs.to_call)
        return a

    def hand_end(self, result):
        try:
            self.mod.on_hand_end(result, self.memory)
        except Exception:
            self.errors += 1


def play_duplicate(spec_a, spec_b, seed, deals: int = 1000) -> dict:
    """One duplicate match; returns agent A's paired bb/deal list summary."""
    a, b = _Seat(load_bot(*spec_a)), _Seat(load_bot(*spec_b))
    seed = str(seed)
    nets = [[0] * deals, [0] * deals]  # nets[half][deal] for A
    for half in (0, 1):
        bots = (a, b) if half == 0 else (b, a)
        a_seat = 0 if half == 0 else 1
        for bot in bots:
            bot.new_half()
        for d in range(deals):
            game = Game(make_deal(seed, d), button_for_deal(d), hand_index=d)
            while not game.is_terminal:
                seat = game.to_act
                game.apply(bots[seat].act(game.observation(seat)))
            nets[half][d] = game.result().nets[a_seat]
            for seat in (0, 1):
                bots[seat].hand_end(game.hand_result(seat))
    paired = [(nets[0][d] + nets[1][d]) / BIG_BLIND for d in range(deals)]
    mean = sum(paired) / deals
    se = statistics.stdev(paired) / math.sqrt(deals)
    return {"mean": mean, "se": se, "errors": (a.errors, b.errors)}


def _job(args):
    spec_a, spec_b, seed, deals = args
    return play_duplicate(spec_a, spec_b, seed, deals)


_POOL: ProcessPoolExecutor | None = None


def run_matches(jobs, workers: int = 20) -> list[dict]:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=workers)
    return list(_POOL.map(_job, jobs, chunksize=1))


def _seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--params-a", default=None)
    ap.add_argument("--params-b", default=None)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    pa = json.loads(args.params_a) if args.params_a else None
    pb = json.loads(args.params_b) if args.params_b else None
    seeds = _seeds(args.seeds)
    res = run_matches([((args.dir_a, pa), (args.dir_b, pb), s, args.deals) for s in seeds],
                      args.workers)
    for s, r in zip(seeds, res):
        print(f"seed {s}: {r['mean']:+.3f} bb/deal (se {r['se']:.3f}) errors {r['errors']}")
    m = [r["mean"] for r in res]
    se = statistics.stdev(m) / math.sqrt(len(m)) if len(m) > 1 else res[0]["se"]
    print(f"MEAN {sum(m) / len(m):+.3f} bb/deal over {len(m)} matches (se {se:.3f})")


if __name__ == "__main__":
    main()
