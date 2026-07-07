"""
Utilità condivise dai test: specs minimali registrati nei registry e un
FakeSession che riproduce una sequenza di gamestate predefinita, così da
testare la logica di MeleeEnv senza avviare Dolphin.
"""
from types import SimpleNamespace

import numpy as np
import melee
from gymnasium import spaces

from smash_rl.environment import MeleeEnv
from smash_rl.specs.observations import register_obs, OBS_SPECS
from smash_rl.specs.rewards import register_reward, REWARD_FNS

TEST_OBS_SHAPE = (4,)

# -- specs minimali usati dai test (registrati una sola volta all'import) --

if "test_minimal" not in OBS_SPECS:
    @register_obs("test_minimal", spaces.Box(-1.0, 1.0, TEST_OBS_SHAPE, np.float32))
    def _build_test(gs, ctx):
        me, opp = gs.players[ctx.agent_port], gs.players[ctx.opp_port]
        feats = [me.percent / 300.0, me.stock / 4.0,
                 opp.percent / 300.0, opp.stock / 4.0]
        return np.clip(np.asarray(feats, np.float32), -1.0, 1.0)

if "test_zero" not in REWARD_FNS:
    @register_reward("test_zero")
    def _reward_zero(prev_gs, gs, ctx):
        return 0.0

if "test_one_per_frame" not in REWARD_FNS:
    @register_reward("test_one_per_frame")
    def _reward_one(prev_gs, gs, ctx):
        return 1.0


def make_gs(frame=100, p1_stock=4, p2_stock=4, p1_percent=0.0, p2_percent=0.0):
    """Gamestate finto con i soli campi usati da env e specs di test."""
    players = {1: SimpleNamespace(stock=p1_stock, percent=p1_percent),
               2: SimpleNamespace(stock=p2_stock, percent=p2_percent)}
    return SimpleNamespace(frame=frame, players=players,
                           menu_state=melee.Menu.IN_GAME)


class FakeSession:
    """
    Sostituto di MeleeSession: step() consuma la sequenza `frames` (gamestate,
    None, o eccezioni da lanciare) senza toccare Dolphin. Replica la logica di
    match_over/stocks/percents della sessione vera.
    """

    def __init__(self, frames):
        self.config = SimpleNamespace(instance_id=0)
        self.frames = list(frames)
        self.old_stocks = [4, 4]
        self._gamestate = None
        self.closed = False
        self.inputs = []

    def step(self):
        gs = self.frames.pop(0) if self.frames else None
        if isinstance(gs, Exception):
            raise gs
        if gs is not None:
            self._gamestate = gs
        return gs

    def apply_input(self, **kwargs):
        self.inputs.append(kwargs)

    @property
    def match_over(self):
        gs = self._gamestate
        if gs is None:
            return False
        stocks = [gs.players[1].stock, gs.players[2].stock]
        a = any(s <= 0 for s in stocks)
        b = any(s > old for s, old in zip(stocks, self.old_stocks))
        self.old_stocks = stocks
        return a or b

    @property
    def stocks(self):
        if self._gamestate is None:
            return np.array([])
        return np.array([self._gamestate.players[1].stock,
                         self._gamestate.players[2].stock])

    @property
    def percents(self):
        if self._gamestate is None:
            return np.array([])
        return np.array([self._gamestate.players[1].percent,
                         self._gamestate.players[2].percent])

    def force_kill_dolphin(self):
        pass

    def close(self):
        self.closed = True


def make_test_env(frames, reward_function="test_zero"):
    """MeleeEnv con specs di test e FakeSession al posto della sessione vera."""
    cfg = SimpleNamespace(instance_id=0)  # MeleeSession si limita a memorizzarla
    env = MeleeEnv(config=cfg,
                   observation_function="test_minimal",
                   action_function="a_only",
                   reward_function=reward_function)
    env.session = FakeSession(frames)
    env._booted_once = True
    return env
