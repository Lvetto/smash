"""
Unit test della logica di MeleeEnv (step, info, truncation, errori recuperabili)
con FakeSession: nessun Dolphin richiesto.
"""
import numpy as np
from melee.slippstream import EnetDisconnected

from smash_rl.tests.helpers import TEST_OBS_SHAPE, make_gs, make_test_env


def test_last_obs_and_space_follow_spec():
    env = make_test_env([])
    assert env.observation_space.shape == TEST_OBS_SHAPE
    assert env._last_obs.shape == TEST_OBS_SHAPE, \
        "_last_obs deve avere la shape dello spec scelto, non quella hardcoded (32,)"


def test_step_counts_decisions_and_truncates():
    frames = [make_gs(frame=i) for i in range(100)]
    env = make_test_env(frames)
    env.max_steps = 2

    _, _, terminated, truncated, _ = env.step(0)
    assert env._steps == 1
    assert not terminated and not truncated

    _, _, terminated, truncated, _ = env.step(0)
    assert env._steps == 2
    assert not terminated
    assert truncated, "raggiunto max_steps l'episodio deve essere troncato"


def test_info_present_when_match_ends_on_first_frame():
    # il match finisce sul primo frame valido dello skip: info deve esserci comunque
    env = make_test_env([make_gs(p2_stock=0, p1_percent=42.0)])

    obs, reward, terminated, truncated, info = env.step(0)

    assert terminated and not truncated
    assert reward == env.win_bonus  # l'avversario è a 0 stock: vittoria
    assert info["P1_stocks"] == 4 and info["P2_stocks"] == 0
    assert info["P1_percent"] == 42.0
    assert obs.shape == TEST_OBS_SHAPE


def test_info_fallback_when_no_valid_frame():
    env = make_test_env([None, None, None])
    env._last_obs = np.full(TEST_OBS_SHAPE, 0.5, np.float32)

    obs, reward, terminated, truncated, info = env.step(0)

    assert not terminated and not truncated
    assert obs is env._last_obs  # stato invariato
    assert info == {"P1_stocks": 0, "P2_stocks": 0, "P1_percent": 0, "P2_percent": 0}


def test_reward_accumulated_once_per_valid_frame():
    # 2 frame validi + 1 None nello skip: il reward per-frame va sommato 2 volte
    frames = [make_gs(frame=1), None, make_gs(frame=2)]
    env = make_test_env(frames, reward_function="test_one_per_frame")

    _, reward, _, _, _ = env.step(0)

    assert reward == 2.0


def test_recoverable_error_truncates_episode():
    env = make_test_env([EnetDisconnected("dolphin morto a metà partita")])
    env._last_obs = np.full(TEST_OBS_SHAPE, 0.5, np.float32)

    obs, reward, terminated, truncated, info = env.step(0)

    assert truncated and not terminated  # truncated=True fa scattare l'auto-reset del VecEnv
    assert info.get("timeout") is True
    assert obs is env._last_obs
    assert env.session.closed, "la sessione va chiusa per forzare il reboot al reset"


def test_action_decode_reaches_controller():
    frames = [make_gs(frame=i) for i in range(10)]
    env = make_test_env(frames)

    env.step(0)  # azione 0 = stick neutro, nessun bottone

    assert len(env.session.inputs) == env.frame_skip
    first = env.session.inputs[0]
    assert (first["stick_x"], first["stick_y"]) == (0.5, 0.5)
    assert first["button"] is None
    assert first["press"] is True
    assert all(not inp["press"] for inp in env.session.inputs[1:]), \
        "il fronte di salita del bottone deve esserci solo sul primo frame dello skip"
