"""
Unit test di make_env: verifica il comportamento in-process (niente redirect,
niente reset anticipato) senza avviare Dolphin. Il boot vero è coperto dal
test marcato `dolphin`.
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

import smash_rl.factory as factory
from smash_rl.factory import make_env
from smash_rl.tests.helpers import TEST_OBS_SHAPE


class DummyEnv(gym.Env):
    observation_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self):
        self.reset_calls = 0
        self.close_calls = 0
        self.kwargs = None

    def close(self):
        self.close_calls += 1

    def reset(self, *, seed=None, options=None):
        self.reset_calls += 1
        return np.zeros(2, np.float32), {}

    def step(self, action):
        return np.zeros(2, np.float32), 0.0, False, False, {}


@pytest.fixture
def patched_factory(monkeypatch):
    """Sostituisce MeleeEnv e MeleeConfig dentro factory con dei finti."""
    dummy = DummyEnv()

    def fake_melee_env(config=None, **kwargs):
        dummy.kwargs = kwargs
        return dummy

    monkeypatch.setattr(factory, "MeleeEnv", fake_melee_env)
    monkeypatch.setattr(
        factory, "MeleeConfig",
        SimpleNamespace(for_instance=lambda i, save_name: SimpleNamespace(
            slippi_port=51441 + i, instance_id=i)),
    )
    return dummy


def test_make_env_in_process(patched_factory):
    stdout, stderr = sys.stdout, sys.stderr

    env = make_env(0, seed=42, save_name="test_save", worker_mode=False,
                   opp_level=3)()

    assert sys.stdout is stdout and sys.stderr is stderr, \
        "con worker_mode=False stdout/stderr non vanno rediretti sul file di log"
    assert isinstance(env, Monitor)
    assert env.unwrapped is patched_factory
    assert patched_factory.kwargs == {"opp_level": 3}  # i kwargs arrivano a MeleeEnv
    assert patched_factory.reset_calls == 0, \
        "make_env non deve chiamare reset(): il boot avviene al primo reset del VecEnv"


def test_make_env_creation_error_propagates(patched_factory, monkeypatch):
    def boom(i, save_name):
        raise FileNotFoundError("ISO non trovata")

    monkeypatch.setattr(factory, "MeleeConfig", SimpleNamespace(for_instance=boom))

    with pytest.raises(FileNotFoundError):
        make_env(0, seed=42, save_name="test_save", worker_mode=False)()


def test_make_env_worker_mode_redirects_and_arms_sigterm(patched_factory, monkeypatch):
    """worker_mode=True: stdout/stderr sul log per istanza e SIGTERM che chiude l'env."""
    import os
    import signal

    exits = []
    monkeypatch.setattr(factory.os, "_exit", lambda code: exits.append(code))

    real_out, real_err = sys.stdout, sys.stderr
    old_handler = signal.getsignal(signal.SIGTERM)
    try:
        make_env(991, seed=0, save_name="test_save", worker_mode=True)()

        assert sys.stdout is sys.stderr
        assert sys.stdout.name == "/tmp/melee_worker_991.log", \
            "ogni worker deve loggare sul proprio file, separato per istanza"

        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler) and handler is not old_handler, \
            "il worker deve installare un handler SIGTERM (per la morte del main)"
        handler(signal.SIGTERM, None)
        assert exits == [1], "alla morte del main il worker deve uscire subito"
        assert patched_factory.close_calls == 1, \
            "l'handler SIGTERM deve chiudere l'env (e quindi Dolphin)"
    finally:
        log = sys.stdout
        sys.stdout, sys.stderr = real_out, real_err
        log.close()
        signal.signal(signal.SIGTERM, old_handler)
        try:
            os.remove("/tmp/melee_worker_991.log")
        except OSError:
            pass


@pytest.mark.dolphin
def test_make_env_boots_dolphin():
    """Integrazione: boot reale di Dolphin con gli specs minimali di test."""
    env = make_env(0, seed=42, save_name="test_save", worker_mode=False,
                   observation_function="test_minimal",
                   action_function="a_only",
                   reward_function="test_zero")()
    try:
        obs, info = env.reset()
        assert obs.shape == TEST_OBS_SHAPE
    finally:
        env.close()
