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


@pytest.mark.parametrize("name", sorted(OBS_SPECS))
def test_every_obs_spec_accepts_gs_and_ctx(name):
    # regressione: ogni build va chiamata come build(gs, ctx) (2 argomenti) —
    # una firma sbagliata crasha in _do_reset e blocca l'intero run in un hang
    space, build = OBS_SPECS[name]
    gs = make_gs()
    ctx = _ctx_with_gamestate(gs)
    if name == "full_v1":                       # stub deprecato: deve fallire chiaro
        with pytest.raises(NotImplementedError):
            build(gs, ctx)
        return
    obs = build(gs, ctx)
    assert obs.shape == space.shape, f"{name}: shape {obs.shape} != {space.shape}"


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


def test_full_obs_observation():
    space, build = OBS_SPECS["full_obs"]
    assert space.shape == (30,)

    gs = make_gs(
        p1_pos=(0.0, 0.0), p2_pos=(40.0, 25.0),
        p1_vel=(1.0, 1.0, 2.5, 2.5), p2_vel=(0.0, 0.0, 0.0, 0.0),
        p1_percent=75.0, p2_percent=450.0,  # 450/150=3.0 -> deve essere clippato a 2.0
        p1_stock=2, p2_stock=4,
        p1_facing=True, p2_facing=False,
        p1_hitstun=25, p2_hitstun=100,
        p1_jumps=1, p2_jumps=2,
        p1_on_ground=True, p2_on_ground=False,
        p1_off_stage=False, p2_off_stage=True,
        p1_invulnerable=False, p2_invulnerable=True,
    )
    obs = build(gs, _ctx_with_gamestate(gs))

    expected = [
        0.0, 0.0,                 # pos agente
        0.5, 0.5, 0.5, 0.5,       # vel agente
        0.5, 0.5, 1.0,            # percent, stock, facing agente
        0.4, 0.5,                 # pos avversario (40/100, 25/50)
        0.0, 0.0, 0.0, 0.0,       # vel avversario
        2.0, 1.0, -1.0,           # percent (clippato da 3.0), stock, facing avversario
        0.4, 0.5,                 # dx, dy relativi (pos avversario - pos agente, dx col segno del facing agente)
        0.5, 2.0,                 # hitstun agente, avversario
        0.5, 1.0,                 # salti rimanenti agente, avversario
        1.0, 0.0,                 # on_ground agente, avversario
        0.0, 1.0,                 # off_stage agente, avversario
        0.0, 1.0,                 # invulnerable agente, avversario
    ]

    assert obs.shape == space.shape and obs.dtype == np.float32
    np.testing.assert_allclose(obs, expected, atol=1e-6)


def test_full_obs_dx_sign_follows_agent_facing():
    space, build = OBS_SPECS["full_obs"]  # default: p1_pos=(0,0), p2_pos=(10,0) -> dx grezzo = 0.1

    gs_right = make_gs(p1_facing=True)
    obs_right = build(gs_right, _ctx_with_gamestate(gs_right))
    assert obs_right[18] == pytest.approx(0.1)
    assert obs_right[19] == pytest.approx(0.0)

    gs_left = make_gs(p1_facing=False)
    obs_left = build(gs_left, _ctx_with_gamestate(gs_left))
    assert obs_left[18] == pytest.approx(-0.1)


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


def test_reward_v2_damage_and_stocks():
    reward = REWARD_FNS["v2"]
    ctx = Ctx(agent_port=1, opp_port=2)
    prev = make_gs()

    # v2 pesa il danno 0.001/punto (10x meno di v1) e lo stock 1.0
    assert reward(prev, make_gs(p2_percent=10.0), ctx) == pytest.approx(0.01)
    assert reward(prev, make_gs(p1_percent=10.0), ctx) == pytest.approx(-0.01)

    # stock tolto: +1; il reset della percentuale a 0 non conta come danno subito
    prev_high = make_gs(p2_percent=120.0)
    assert reward(prev_high, make_gs(p2_stock=3, p2_percent=0.0), ctx) == pytest.approx(1.0)

    # stock perso: -1
    assert reward(prev, make_gs(p1_stock=3, p1_percent=0.0), ctx) == pytest.approx(-1.0)


def test_reward_v2_stock_dominates_damage():
    reward = REWARD_FNS["v2"]
    ctx = Ctx(agent_port=1, opp_port=2)

    # una kill (+1) vale più di qualunque danno subito nello stesso frame (max ~ -0.001*danno):
    # è proprio il senso di v2 rispetto a v1
    prev = make_gs()
    gs = make_gs(p2_stock=3, p1_percent=200.0)  # tolgo uno stock ma incasso 200 di danno
    assert reward(prev, gs, ctx) == pytest.approx(1.0 - 0.001 * 200)
    assert reward(prev, gs, ctx) > 0.0


def test_reward_v2_is_symmetric():
    reward = REWARD_FNS["v2"]
    ctx = Ctx(agent_port=1, opp_port=2)
    ctx_swapped = Ctx(agent_port=2, opp_port=1)

    prev = make_gs()
    gs = make_gs(p1_percent=10.0, p2_percent=25.0, p1_stock=3)
    assert reward(prev, gs, ctx) == pytest.approx(-reward(prev, gs, ctx_swapped))


def test_reward_v2_edge_cases():
    reward = REWARD_FNS["v2"]
    ctx = Ctx(agent_port=1, opp_port=2)

    # niente reward senza stato precedente o durante il countdown
    assert reward(None, make_gs(p2_percent=50.0), ctx) == 0.0
    assert reward(make_gs(), make_gs(frame=-20, p2_percent=50.0), ctx) == 0.0

    # stock aumentato = respawn/restart, non una kill
    assert reward(make_gs(p2_stock=1), make_gs(p2_stock=4), ctx) == 0.0


def test_reward_v3_equals_v2_without_attack_edge():
    v2, v3 = REWARD_FNS["v2"], REWARD_FNS["v3"]
    ctx = Ctx(agent_port=1, opp_port=2)

    # nessuna transizione d'attacco (entrambi STANDING): v3 == v2, danni/stock inclusi
    prev = make_gs()
    gs = make_gs(p2_percent=10.0, p1_stock=3)
    assert v3(prev, gs, ctx) == pytest.approx(v2(prev, gs, ctx))

    # attacco che continua (prev e curr entrambi NAIR): nessun edge, niente penalità extra
    prev_atk = make_gs(p1_action=melee.Action.NAIR)
    gs_atk = make_gs(p1_action=melee.Action.NAIR, p2_percent=10.0)
    assert v3(prev_atk, gs_atk, ctx) == pytest.approx(v2(prev_atk, gs_atk, ctx))


def test_reward_v3_penalizes_attack_start():
    v2, v3 = REWARD_FNS["v2"], REWARD_FNS["v3"]
    ctx = Ctx(agent_port=1, opp_port=2)

    # l'agente entra in un attacco (STANDING -> NAIR): v3 = v2 - 0.001
    prev = make_gs(p1_action=melee.Action.STANDING)
    gs = make_gs(p1_action=melee.Action.NAIR)
    assert v3(prev, gs, ctx) == pytest.approx(v2(prev, gs, ctx) - 0.001)

    # la penalità si somma al reward di danno dello stesso frame
    gs_dmg = make_gs(p1_action=melee.Action.NAIR, p2_percent=10.0)
    assert v3(prev, gs_dmg, ctx) == pytest.approx(0.01 - 0.001)

    # anche un grab conta come attacco (grab inclusi)
    gs_grab = make_gs(p1_action=melee.Action.GRAB)
    assert v3(prev, gs_grab, ctx) == pytest.approx(v2(prev, gs_grab, ctx) - 0.001)


def test_reward_v3_ignores_opponent_attack():
    v2, v3 = REWARD_FNS["v2"], REWARD_FNS["v3"]
    ctx = Ctx(agent_port=1, opp_port=2)

    # un attacco dell'avversario non penalizza l'agente
    prev = make_gs(p2_action=melee.Action.STANDING)
    gs = make_gs(p2_action=melee.Action.NAIR)
    assert v3(prev, gs, ctx) == pytest.approx(v2(prev, gs, ctx))


def test_reward_v3_edge_cases():
    v3 = REWARD_FNS["v3"]
    ctx = Ctx(agent_port=1, opp_port=2)

    # niente reward (né penalità) senza stato precedente o durante il countdown,
    # anche se il frame corrente è già in un attacco
    assert v3(None, make_gs(p1_action=melee.Action.NAIR), ctx) == 0.0
    assert v3(make_gs(), make_gs(frame=-20, p1_action=melee.Action.NAIR), ctx) == 0.0


def test_session_properties_from_gamestate():
    session = FakeSession([])  # eredita le properties vere di MeleeSession

    # senza gamestate: array vuoti e distanza nulla, niente eccezioni
    assert len(session.positions) == 0
    assert len(session.velocities) == 0
    assert len(session.stocks) == 0
    assert len(session.percents) == 0
    assert len(session.facings) == 0
    assert len(session.hitstun_frames) == 0
    assert len(session.jumps_left) == 0
    assert len(session.on_ground) == 0
    assert len(session.off_stage) == 0
    assert len(session.invulnerable) == 0
    assert session.distance == 0.0

    session._gamestate = make_gs(p1_pos=(10.0, 5.0), p2_pos=(-10.0, 0.0),
                                 p1_vel=(1.0, 2.0, 3.0, 4.0),
                                 p1_percent=42.0, p2_stock=2, distance=30.0,
                                 p1_facing=True, p2_facing=False,
                                 p1_hitstun=15, p2_hitstun=0,
                                 p1_jumps=0, p2_jumps=2,
                                 p1_on_ground=False, p2_on_ground=True,
                                 p1_off_stage=True, p2_off_stage=False,
                                 p1_invulnerable=True, p2_invulnerable=False)

    np.testing.assert_allclose(session.positions, [[10.0, 5.0], [-10.0, 0.0]])
    np.testing.assert_allclose(session.velocities[0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(session.stocks, [4, 2])
    np.testing.assert_allclose(session.percents, [42.0, 0.0])
    assert session.distance == 30.0
    np.testing.assert_allclose(session.facings, [1.0, -1.0])
    np.testing.assert_allclose(session.hitstun_frames, [15, 0])
    np.testing.assert_allclose(session.jumps_left, [0, 2])
    np.testing.assert_allclose(session.on_ground, [0.0, 1.0])
    np.testing.assert_allclose(session.off_stage, [1.0, 0.0])
    np.testing.assert_allclose(session.invulnerable, [1.0, 0.0])
