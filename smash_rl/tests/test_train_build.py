"""
Unit test di train.run(): costruzione del modello giusto con i kwargs
pass-through e gestione del resume, su un DummyVecEnv senza Dolphin.
"""
import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import smash_rl.train as train
from smash_rl.train import TrainConfig


class DummyEnv(gym.Env):
    observation_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
    action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return np.zeros(2, np.float32), {}

    def step(self, action):
        return np.zeros(2, np.float32), 0.0, False, False, {}


@pytest.fixture
def patched_train(monkeypatch, tmp_path):
    """run() su DummyEnv, con cwd su tmp_path (tb_logs/ e checkpoints/ finti)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "make_env", lambda *a, **kw: DummyEnv)
    monkeypatch.setattr(train, "SubprocVecEnv",
                        lambda fns, start_method=None: DummyVecEnv(fns))
    return tmp_path


def _cfg(**changes):
    base = dict(run_name="t", algo="dqn", n_envs=2, total_steps=8, ckpt_every=0,
                boot_stagger_s=0.0, instance_base=0,
                algo_kwargs=dict(buffer_size=100, learning_starts=1000),
                env_kwargs={})
    base.update(changes)
    return TrainConfig(**base)


def test_run_builds_algo_with_passthrough_kwargs(patched_train, monkeypatch):
    created = {}

    class SpyDQN(DQN):
        def __init__(self, *args, **kwargs):
            created.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setitem(train.ALGOS, "dqn", SpyDQN)

    train.run(_cfg())

    assert created["buffer_size"] == 100, "gli algo_kwargs devono arrivare al costruttore SB3"
    assert created["learning_starts"] == 1000
    assert (patched_train / "checkpoints" / "t" / "final.zip").exists(), \
        "a fine run i pesi vanno salvati in checkpoints/<run>/final.zip"


def test_run_requires_instance_base(patched_train):
    with pytest.raises(ValueError, match="instance_base"):
        train.run(_cfg(instance_base=None))


def _save_tiny_dqn(path):
    model = DQN("MlpPolicy", DummyVecEnv([DummyEnv]), buffer_size=100,
                exploration_initial_eps=1.0, exploration_final_eps=0.02)
    model.save(path)
    return model


def test_run_resume_applies_exploration_reset_only_if_configured(patched_train, monkeypatch):
    pretrained = patched_train / "pretrained.zip"
    _save_tiny_dqn(pretrained)

    loaded = {}

    class SpyDQN(DQN):
        @classmethod
        def load(cls, *args, **kwargs):
            model = super().load(*args, **kwargs)
            loaded["model"] = model
            return model

    monkeypatch.setitem(train.ALGOS, "dqn", SpyDQN)

    # senza reset_exploration: lo schedule resta quello salvato
    train.run(_cfg(pretrained_path=str(pretrained)))
    assert loaded["model"].exploration_initial_eps == 1.0

    # con reset_exploration: lo schedule viene riscritto
    train.run(_cfg(run_name="t2", pretrained_path=str(pretrained),
                   reset_exploration={"initial_eps": 0.3, "final_eps": 0.05,
                                      "fraction": 0.3}))
    model = loaded["model"]
    assert model.exploration_initial_eps == 0.3
    assert model.exploration_final_eps == 0.05
    assert model.exploration_schedule(1.0) == pytest.approx(0.3), \
        "a inizio training l'esplorazione deve partire dal nuovo initial_eps"
