"""
Unit test di TrainConfig: serializzazione, validazione e override da CLI,
senza avviare Dolphin né costruire modelli.
"""
import dataclasses

import melee
import pytest

from smash_rl.train import TrainConfig, _parse_args, resolve_env_kwargs


def test_json_round_trip(tmp_path):
    cfg = TrainConfig(run_name="rt", algo="ppo", n_envs=2, total_steps=123,
                      algo_kwargs={"n_steps": 8}, instance_base=5,
                      env_kwargs={"agent_char": "FOX", "opp_level": 3})
    path = tmp_path / "cfg.json"
    cfg.to_json(path)
    assert TrainConfig.from_json(path) == cfg


def test_from_json_rejects_unknown_fields(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text('{"algo": "dqn", "totale_steps": 5}')
    with pytest.raises(ValueError, match="totale_steps"):
        TrainConfig.from_json(path)


def test_resolve_env_kwargs_converts_enums():
    cfg = TrainConfig(env_kwargs={"agent_char": "FOX", "opp_char": "MARTH",
                                  "stage": "BATTLEFIELD", "opp_level": 7})
    kwargs = resolve_env_kwargs(cfg)
    assert kwargs["agent_char"] is melee.Character.FOX
    assert kwargs["opp_char"] is melee.Character.MARTH
    assert kwargs["stage"] is melee.Stage.BATTLEFIELD
    assert kwargs["opp_level"] == 7
    # la config originale resta con le stringhe (deve rimanere serializzabile)
    assert cfg.env_kwargs["agent_char"] == "FOX"


@pytest.mark.parametrize("changes,match", [
    (dict(algo="dddqn"), "algo"),
    (dict(env_kwargs={"observation_function": "boh"}), "observation_function"),
    (dict(env_kwargs={"action_function": "boh"}), "action_function"),
    (dict(env_kwargs={"reward_function": "boh"}), "reward_function"),
    (dict(env_kwargs={"agent_char": "PIPPO"}), "agent_char"),
    (dict(algo="ppo", reset_exploration={"initial_eps": 0.3, "final_eps": 0.05,
                                         "fraction": 0.3}), "dqn"),
    (dict(reset_exploration={"initial_eps": 0.3}), "final_eps"),
    (dict(n_envs=0), "n_envs"),
])
def test_validate_rejects(changes, match):
    cfg = dataclasses.replace(TrainConfig(), **changes)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_validate_accepts_default():
    TrainConfig().validate()


def test_cli_overrides_win_over_file(tmp_path):
    path = tmp_path / "cfg.json"
    TrainConfig(run_name="dal_file", n_envs=4, total_steps=1000).to_json(path)

    cfg = _parse_args(["--config", str(path), "--run-name", "da_cli",
                       "--n-envs", "2", "--instance-base", "8", "--algo", "ppo"])

    assert cfg.run_name == "da_cli"
    assert cfg.n_envs == 2
    assert cfg.instance_base == 8
    assert cfg.algo == "ppo"
    assert cfg.total_steps == 1000, "i campi non passati da CLI restano quelli del file"
