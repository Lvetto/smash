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
from smash_rl.session import MeleeSession
from smash_rl.specs.context import Ctx
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


def make_gs(frame=100, p1_stock=4, p2_stock=4, p1_percent=0.0, p2_percent=0.0,
            p1_pos=(0.0, 0.0), p2_pos=(10.0, 0.0),
            p1_vel=(0.0, 0.0, 0.0, 0.0), p2_vel=(0.0, 0.0, 0.0, 0.0),
            p1_facing=True, p2_facing=False,
            p1_hitstun=0, p2_hitstun=0,
            p1_jumps=1, p2_jumps=1,
            p1_on_ground=True, p2_on_ground=True,
            p1_off_stage=False, p2_off_stage=False,
            p1_invulnerable=False, p2_invulnerable=False,
            distance=10.0):
    """
    Gamestate finto con i campi usati da env e specs. Le velocità sono tuple
    (vx_self, vy_self, vx_attack, vy_attack); vx_self finisce nella componente a terra.
    facing è booleano (True = rivolto a destra), come in libmelee.
    """
    def player(stock, percent, pos, vel, facing, hitstun, jumps, on_ground, off_stage, invulnerable):
        return SimpleNamespace(
            stock=stock, percent=percent, facing=facing,
            position=SimpleNamespace(x=pos[0], y=pos[1]),
            speed_ground_x_self=vel[0], speed_air_x_self=0.0,
            speed_y_self=vel[1], speed_x_attack=vel[2], speed_y_attack=vel[3],
            hitstun_frames_left=hitstun, jumps_left=jumps,
            on_ground=on_ground, off_stage=off_stage, invulnerable=invulnerable,
        )

    players = {1: player(p1_stock, p1_percent, p1_pos, p1_vel, p1_facing,
                        p1_hitstun, p1_jumps, p1_on_ground, p1_off_stage, p1_invulnerable),
               2: player(p2_stock, p2_percent, p2_pos, p2_vel, p2_facing,
                        p2_hitstun, p2_jumps, p2_on_ground, p2_off_stage, p2_invulnerable)}
    return SimpleNamespace(frame=frame, players=players, distance=distance,
                           menu_state=melee.Menu.IN_GAME)


class FakeSession(MeleeSession):
    """
    Sostituto di MeleeSession: step() consuma la sequenza `frames` (gamestate,
    None, o eccezioni da lanciare) senza toccare Dolphin. Eredita le properties
    vere (positions, velocities, stocks, percents, distance, match_over), quindi
    i test le esercitano davvero.
    """

    def __init__(self, frames):
        # niente super().__init__(): nessuna config reale, nessun Dolphin
        self.config = SimpleNamespace(instance_id=0)
        self.players = [None, None]  # serve solo len() nelle properties
        self.old_stocks = [4, 4]
        self._gamestate = None
        self._external_console = None
        self.console = None
        self.display = None
        self.frames = list(frames)
        self.closed = False
        self.close_calls = 0
        self.hard_resets = 0
        self.inputs = []

    def hard_reset(self):
        self.hard_resets += 1
        return self

    def advance_to_in_game(self, **kwargs):
        return None  # il primo elemento di `frames` fa da gamestate d'ingresso in partita

    def step(self):
        gs = self.frames.pop(0) if self.frames else None
        if isinstance(gs, Exception):
            raise gs
        if gs is not None:
            self._gamestate = gs
        return gs

    def apply_input(self, **kwargs):
        self.inputs.append(kwargs)

    def force_kill_dolphin(self):
        pass

    def close(self):
        self.closed = True
        self.close_calls += 1


def make_test_env(frames, observation_function="test_minimal", reward_function="test_zero"):
    """MeleeEnv con specs di test e FakeSession al posto della sessione vera."""
    cfg = SimpleNamespace(instance_id=0)  # MeleeSession si limita a memorizzarla
    env = MeleeEnv(config=cfg,
                   observation_function=observation_function,
                   action_function="a_only",
                   reward_function=reward_function)
    env.session = FakeSession(frames)
    env.ctx = Ctx(agent_port=1, opp_port=2, session=env.session)  # il ctx deve puntare alla sessione finta
    env._booted_once = True
    return env
