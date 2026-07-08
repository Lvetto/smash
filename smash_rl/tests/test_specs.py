"""Unit test dei registry di specs (azioni, osservazioni, reward)."""
import melee
import numpy as np
import pytest

from smash_rl.specs import ACT_SPECS, OBS_SPECS, REWARD_FNS, Ctx
from smash_rl.specs.actions import STICK_MAP, BUTTONS_FULL, BUTTONS_A_ONLY
from smash_rl.tests.helpers import FakeSession, make_gs


def _ctx_with_gamestate(gs):
    session = FakeSession([])
    session._gamestate = gs
    return Ctx(agent_port=1, opp_port=2, session=session)


def test_full_action_spec_covers_all_combinations():
    space, decode = ACT_SPECS["full"]
    assert space.n == len(STICK_MAP) * len(BUTTONS_FULL)  # 54

    seen = set()
    for action in range(space.n):
        (x, y), button = decode(action)
        assert (x, y) in STICK_MAP.values()
        assert button in BUTTONS_FULL.values()
        seen.add(((x, y), button))
    assert len(seen) == space.n, "ogni azione deve decodificare in una combinazione unica"


def test_a_only_action_spec():
    space, decode = ACT_SPECS["a_only"]
    assert space.n == len(STICK_MAP) * len(BUTTONS_A_ONLY)  # 18

    buttons = {decode(action)[1] for action in range(space.n)}
    assert buttons == {None, melee.Button.BUTTON_A}


def test_neutral_action_is_zero():
    _, decode = ACT_SPECS["full"]
    assert decode(0) == ((0.5, 0.5), None)


def test_registries_expose_default_specs():
    assert "full_v1" in OBS_SPECS
    space, build = OBS_SPECS["full_v1"]
    assert space.shape == (32,)
    with pytest.raises(NotImplementedError):  # stub: deve fallire chiaro, non ritornare None
        build(make_gs(), _ctx_with_gamestate(make_gs()))

    assert "v1" in REWARD_FNS


def test_pos_vel_observation():
    space, build = OBS_SPECS["pos_vel"]
    assert space.shape == (12,)

    gs = make_gs(p1_pos=(42.5, 25.0), p2_pos=(-85.0, 50.0),
                 p1_vel=(2.5, -5.0, 0.0, 0.0))
    obs = build(gs, _ctx_with_gamestate(gs))

    assert obs.shape == space.shape and obs.dtype == np.float32
    np.testing.assert_allclose(obs[0:2], [0.5, 0.5])             # pos agente / (85, 50)
    np.testing.assert_allclose(obs[2:6], [0.5, -1.0, 0.0, 0.0])  # vel agente / 5
    np.testing.assert_allclose(obs[6:8], [-1.0, 1.0])            # pos avversario


def test_pos_vel_stats_observation():
    space, build = OBS_SPECS["pos_vel_stats"]
    assert space.shape == (17,)

    gs = make_gs(p1_percent=150.0, p2_percent=600.0, p1_stock=4, p2_stock=1,
                 distance=50.0)
    obs = build(gs, _ctx_with_gamestate(gs))

    assert obs.shape == space.shape and obs.dtype == np.float32
    # danni/300 e vite/4; il 600% dell'avversario va oltre il range e viene clippato a 1
    np.testing.assert_allclose(obs[12:16], [0.5, 1.0, 1.0, 0.25])
    np.testing.assert_allclose(obs[16], 0.5)  # distanza / 100


def test_reward_v1_damage_and_stocks():
    reward = REWARD_FNS["v1"]
    ctx = Ctx(agent_port=1, opp_port=2)
    prev = make_gs()

    # danno inflitto: +0.01 a punto; danno subito: -0.01
    assert reward(prev, make_gs(p2_percent=10.0), ctx) == pytest.approx(0.1)
    assert reward(prev, make_gs(p1_percent=10.0), ctx) == pytest.approx(-0.1)

    # stock tolto: +1; il reset della percentuale a 0 non conta come danno subito
    prev_high = make_gs(p2_percent=120.0)
    assert reward(prev_high, make_gs(p2_stock=3, p2_percent=0.0), ctx) == pytest.approx(1.0)

    # stock perso: -1
    assert reward(prev, make_gs(p1_stock=3, p1_percent=0.0), ctx) == pytest.approx(-1.0)


def test_reward_v1_is_symmetric():
    reward = REWARD_FNS["v1"]
    ctx = Ctx(agent_port=1, opp_port=2)
    ctx_swapped = Ctx(agent_port=2, opp_port=1)

    prev = make_gs()
    gs = make_gs(p1_percent=10.0, p2_percent=25.0, p1_stock=3)
    assert reward(prev, gs, ctx) == pytest.approx(-reward(prev, gs, ctx_swapped))


def test_reward_v1_edge_cases():
    reward = REWARD_FNS["v1"]
    ctx = Ctx(agent_port=1, opp_port=2)

    # niente reward senza stato precedente o durante il countdown
    assert reward(None, make_gs(p2_percent=50.0), ctx) == 0.0
    assert reward(make_gs(), make_gs(frame=-20, p2_percent=50.0), ctx) == 0.0

    # stock aumentato = respawn/restart, non una kill
    assert reward(make_gs(p2_stock=1), make_gs(p2_stock=4), ctx) == 0.0


def test_session_properties_from_gamestate():
    session = FakeSession([])  # eredita le properties vere di MeleeSession

    # senza gamestate: array vuoti e distanza nulla, niente eccezioni
    assert len(session.positions) == 0
    assert len(session.velocities) == 0
    assert len(session.stocks) == 0
    assert len(session.percents) == 0
    assert session.distance == 0.0

    session._gamestate = make_gs(p1_pos=(10.0, 5.0), p2_pos=(-10.0, 0.0),
                                 p1_vel=(1.0, 2.0, 3.0, 4.0),
                                 p1_percent=42.0, p2_stock=2, distance=30.0)

    np.testing.assert_allclose(session.positions, [[10.0, 5.0], [-10.0, 0.0]])
    np.testing.assert_allclose(session.velocities[0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(session.stocks, [4, 2])
    np.testing.assert_allclose(session.percents, [42.0, 0.0])
    assert session.distance == 30.0
