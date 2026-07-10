"""
Integrazione end-to-end del launcher (pytest -m dolphin): avvia veri training
in background con Dolphin e verifica ciclo di vita e stop. Usa la RUNS_DIR
reale (è il percorso davvero esercitato); i run hanno nomi pytest_* usa-e-getta.
"""
import os
import time

import psutil
import pytest

import smash_rl.runs as runs
from smash_rl.train import TrainConfig

pytestmark = pytest.mark.dolphin


def _tiny_cfg(run_name, total_steps):
    return TrainConfig(
        run_name=run_name, algo="dqn", n_envs=1, total_steps=total_steps,
        ckpt_every=0, boot_stagger_s=0.0,
        algo_kwargs=dict(buffer_size=1_000, learning_starts=50),
        env_kwargs=dict(agent_char="FOX", opp_char="MARTH", opp_level=3,
                        observation_function="pos_vel", action_function="a_only",
                        reward_function="v1"),
    )


def _wait_status(run_name, wanted, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = runs.status(run_name)
        if st["status"] in wanted:
            return st
        time.sleep(5)
    pytest.fail(f"{run_name} non è arrivato a {wanted} entro {timeout_s}s; "
                f"ultimo log: {runs.tail_log(run_name, n=5)}")


def _dolphin_count():
    return sum(1 for p in psutil.process_iter(["name"])
               if p.info["name"] and "dolphin-emu" in p.info["name"])


def test_launch_runs_to_completion():
    run_name = f"pytest_dolphin_{os.getpid()}"
    h = runs.launch(_tiny_cfg(run_name, total_steps=400))

    st = _wait_status(run_name, {"finished", "crashed"}, timeout_s=360)
    assert st["status"] == "finished", runs.tail_log(run_name, n=20)
    from smash_rl.runs import REPO_ROOT
    assert (REPO_ROOT / "checkpoints" / run_name / "final.zip").exists()
    assert not psutil.pid_exists(h.pid)


def test_stop_kills_run_and_its_dolphin():
    run_name = f"pytest_stop_{os.getpid()}"
    dolphins_before = _dolphin_count()
    h = runs.launch(_tiny_cfg(run_name, total_steps=1_000_000))

    # aspetta che il worker abbia avviato il suo Dolphin
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline and _dolphin_count() <= dolphins_before:
        time.sleep(3)
    assert _dolphin_count() > dolphins_before, "Dolphin non è mai partito"

    runs.stop(run_name, timeout_s=60.0)

    assert not psutil.pid_exists(h.pid)
    assert runs.status(run_name)["status"] == "stopped"
    time.sleep(5)  # lascia morire l'albero di processi
    assert _dolphin_count() == dolphins_before, \
        "lo stop deve chiudere i Dolphin del run senza toccare gli altri"
