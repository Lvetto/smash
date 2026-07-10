"""
Unit test del registry dei run: allocazione degli slot, riconciliazione dei
processi morti, launch (con Popen finto) e stop (su un subprocess vero ma
innocuo), tutto su una RUNS_DIR temporanea.
"""
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

import smash_rl.runs as runs
from smash_rl.train import TrainConfig


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    return tmp_path / "runs"


def _entry(base, n, status="running", **extra):
    e = dict(run_id=0, run_name=f"r{base}", pid=os.getpid(),
             pid_create_time=psutil.Process().create_time(),
             instance_base=base, n_envs=n, config_path="", log_path="",
             started_at="", status=status, algo="dqn", total_steps=100)
    e.update(extra)
    return e


# -- allocazione slot --

@pytest.mark.parametrize("live,n,expected", [
    ([], 4, 0),                                    # registry vuoto
    ([(0, 4), (4, 2)], 4, 6),                      # accoda dopo gli intervalli vivi
    ([(4, 2)], 4, 0),                              # buco iniziale abbastanza grande
    ([(2, 2)], 4, 4),                              # buco iniziale troppo piccolo
    ([(0, 2), (6, 2)], 4, 2),                      # buco in mezzo
    ([(6, 2), (0, 2)], 4, 2),                      # l'ordine delle entry non conta
])
def test_first_fit_base(live, n, expected):
    entries = [_entry(base, envs) for base, envs in live]
    assert runs._first_fit_base(entries, n) == expected


# -- liveness e riconciliazione --

def test_is_alive_checks_pid_and_create_time():
    assert runs._is_alive(_entry(0, 1))
    assert not runs._is_alive(_entry(0, 1, pid_create_time=1.0)), \
        "un create_time diverso significa che il PID è stato riusato"


def test_is_alive_treats_zombies_as_dead():
    # figlio uscito ma non ancora raccolto (niente wait): è il caso di un run
    # lanciato da un notebook che termina da solo
    proc = subprocess.Popen(["true"])
    ctime = psutil.Process(proc.pid).create_time()
    time.sleep(0.3)  # lascia uscire il processo senza raccoglierlo
    try:
        assert psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE
        assert not runs._is_alive(_entry(0, 1, pid=proc.pid, pid_create_time=ctime))
    finally:
        proc.wait()


def test_reconcile_marks_dead_running_as_crashed(runs_dir):
    dead = subprocess.Popen(["true"])
    dead.wait()
    reg = {"next_id": 2, "runs": {
        "vivo": _entry(0, 2),
        "morto": _entry(2, 2, pid=dead.pid, pid_create_time=1.0),
        "finito": _entry(4, 2, status="finished", pid=dead.pid),
    }}
    runs._reconcile(reg)
    assert reg["runs"]["vivo"]["status"] == "running"
    assert reg["runs"]["morto"]["status"] == "crashed"
    assert reg["runs"]["finito"]["status"] == "finished", \
        "la riconciliazione tocca solo i run 'running'"


# -- launch --

@pytest.fixture
def fake_popen(monkeypatch):
    calls = []

    def popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(pid=os.getpid())  # un pid vivo, così l'entry risulta running

    monkeypatch.setattr(runs.subprocess, "Popen", popen)
    return calls


def test_launch_allocates_and_registers(runs_dir, fake_popen):
    h1 = runs.launch(TrainConfig(n_envs=4, total_steps=100))
    h2 = runs.launch(TrainConfig(n_envs=2, total_steps=100))

    assert (h1.run_name, h2.run_name) == ("run_000", "run_001")
    cfg1 = TrainConfig.from_json(h1.config_path)
    cfg2 = TrainConfig.from_json(h2.config_path)
    assert cfg1.instance_base == 0
    assert cfg2.instance_base == 4, "il secondo run vivo deve avere istanze disgiunte"

    reg = json.loads((runs_dir / "registry.json").read_text())
    assert reg["next_id"] == 2
    assert reg["runs"]["run_000"]["status"] == "running"

    cmd, kwargs = fake_popen[0]
    assert cmd[:3] == [sys.executable, "-m", "smash_rl.train"]
    assert kwargs["start_new_session"], "serve un process group proprio per stop() e per sopravvivere a Jupyter"


def test_launch_reuses_slots_of_dead_runs(runs_dir, fake_popen, monkeypatch):
    runs.launch(TrainConfig(n_envs=4, total_steps=100))
    monkeypatch.setattr(runs, "_is_alive", lambda entry: False)  # run_000 "muore"
    h = runs.launch(TrainConfig(n_envs=4, total_steps=100))
    assert TrainConfig.from_json(h.config_path).instance_base == 0, \
        "gli slot di un run morto vanno riusati"


def test_launch_rejects_live_name_collision(runs_dir, fake_popen):
    runs.launch(TrainConfig(run_name="stesso", n_envs=1, total_steps=100))
    with pytest.raises(ValueError, match="stesso"):
        runs.launch(TrainConfig(run_name="stesso", n_envs=1, total_steps=100))


# -- status / stop / mark_finished --

def test_status_and_stop_on_real_process(runs_dir):
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    with runs._locked_registry() as reg:
        reg["runs"]["s"] = _entry(0, 1, pid=proc.pid,
                                  pid_create_time=psutil.Process(proc.pid).create_time(),
                                  run_name="s")
    assert runs.status("s")["alive"]

    runs.stop("s", timeout_s=10.0)

    assert proc.wait(timeout=5) is not None
    assert runs.status("s")["status"] == "stopped"
    assert runs.list_runs() == [], "un run fermato non è più tra i vivi"
    assert runs.list_runs(all=True)[0]["run_name"] == "s"


def test_stop_escalates_to_killpg(runs_dir):
    # un processo che ignora SIGTERM: serve l'escalation SIGKILL sul process group
    proc = subprocess.Popen([sys.executable, "-c",
                             "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"],
                            start_new_session=True)
    time.sleep(0.5)  # lascia installare il handler
    with runs._locked_registry() as reg:
        reg["runs"]["z"] = _entry(0, 1, pid=proc.pid,
                                  pid_create_time=psutil.Process(proc.pid).create_time(),
                                  run_name="z")

    runs.stop("z", timeout_s=2.0)

    assert proc.wait(timeout=5) == -9
    assert runs.status("z")["status"] == "stopped"


def test_mark_finished_ignores_unregistered_runs(runs_dir):
    runs.mark_finished("mai_visto", "finished")  # non deve alzare eccezioni
    assert runs.list_runs(all=True) == []
