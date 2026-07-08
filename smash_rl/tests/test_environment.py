"""
Unit test della logica di MeleeEnv (step, info, truncation, errori recuperabili)
con FakeSession: nessun Dolphin richiesto.
"""
import numpy as np
import pytest
from melee.slippstream import EnetDisconnected

import smash_rl.environment
from smash_rl.specs.context import Ctx
from smash_rl.tests.helpers import TEST_OBS_SHAPE, FakeSession, make_gs, make_test_env


class FlakySession(FakeSession):
    """FakeSession il cui hard_reset fallisce le prime `fail_times` volte."""

    def __init__(self, frames, fail_times, error=TimeoutError):
        super().__init__(frames)
        self.fail_times = fail_times
        self.error = error

    def hard_reset(self):
        self.hard_resets += 1
        if self.hard_resets <= self.fail_times:
            raise self.error("menu incastrati")
        return self


def _no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(smash_rl.environment.time, "sleep", sleeps.append)
    return sleeps


def test_reset_returns_obs_and_syncs_state():
    frames = [make_gs(frame=5, p1_percent=10.0)]
    env = make_test_env(frames)

    obs, info = env.reset()

    assert env.session.hard_resets == 1
    assert obs.shape == TEST_OBS_SHAPE
    assert info == {}
    assert env._steps == 0
    assert env.session.old_stocks == [4, 4], \
        "old_stocks va riallineato al nuovo match, altrimenti match_over scatta subito"
    assert len(env.session.inputs) >= 1, \
        "in attesa del primo frame va inviato input neutro (con blocking_input Dolphin aspetta)"


def test_reset_waits_boot_delay_only_once(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    env = make_test_env([make_gs(), make_gs()])
    env._booted_once = False
    env.boot_delay_s = 3.3

    env.reset()
    assert 3.3 in sleeps and env._booted_once

    sleeps.clear()
    env.reset()
    assert 3.3 not in sleeps, "il boot delay serve solo al primo avvio"


def test_reset_retries_on_recoverable_error(monkeypatch):
    _no_sleep(monkeypatch)
    env = make_test_env([])
    env.session = FlakySession([make_gs()], fail_times=2)
    env.ctx = Ctx(agent_port=1, opp_port=2, session=env.session)

    obs, _ = env.reset()

    assert env.session.hard_resets == 3
    assert env.session.close_calls == 2, \
        "dopo ogni tentativo fallito la sessione va chiusa per forzare un reboot pulito"
    assert obs.shape == TEST_OBS_SHAPE


def test_reset_gives_up_after_max_attempts(monkeypatch):
    _no_sleep(monkeypatch)
    env = make_test_env([])
    env.session = FlakySession([], fail_times=99, error=RuntimeError)

    with pytest.raises(RuntimeError, match="reset fallito"):
        env.reset()

    assert env.session.hard_resets == env.max_reset_attempts
    assert env.session.close_calls == env.max_reset_attempts


def test_reset_propagates_fatal_errors(monkeypatch):
    _no_sleep(monkeypatch)
    env = make_test_env([])
    env.session = FlakySession([], fail_times=99, error=KeyError)

    with pytest.raises(KeyError):
        env.reset()
    assert env.session.hard_resets == 1, "gli errori non recuperabili non vanno ritentati"


def test_close_closes_session():
    env = make_test_env([])
    env.close()
    assert env.session.closed


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


def test_env_with_session_based_observation():
    # end-to-end con uno spec che legge le properties della session (via ctx.session)
    frames = [make_gs(frame=i, distance=20.0) for i in range(10)]
    env = make_test_env(frames, observation_function="pos_vel_stats")

    obs, _, _, _, _ = env.step(0)

    assert obs.shape == (17,)
    assert env.observation_space.contains(obs)


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
