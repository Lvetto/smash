import dataclasses
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psutil

from smash_rl.train import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
SLIPPI_BASE_PORT = 51441        # deve combaciare con MeleeConfig.for_instance


@dataclass
class RunHandle:
    run_name: str
    run_id: int
    pid: int
    log_path: Path
    config_path: Path


@contextmanager
def _locked_registry():
    """
    Registry sotto flock: carica runs/registry.json, yield del dict (mutabile),
    riscrittura atomica (.tmp + os.replace) al termine. Tutte le funzioni
    pubbliche passano da qui, così allocazioni concorrenti non si pestano.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            reg_path = RUNS_DIR / "registry.json"
            if reg_path.exists():
                reg = json.loads(reg_path.read_text())
            else:
                reg = {"next_id": 0, "runs": {}}
            yield reg
            tmp = reg_path.with_name("registry.json.tmp")
            tmp.write_text(json.dumps(reg, indent=2) + "\n")
            os.replace(tmp, reg_path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _is_alive(entry: dict) -> bool:
    """Il processo del run esiste ancora? Confronta anche il create_time per non farsi ingannare dal riuso dei PID."""
    try:
        proc = psutil.Process(entry["pid"])
        if proc.status() == psutil.STATUS_ZOMBIE:
            # morto ma non ancora raccolto dal padre (tipico se lanciato da un
            # notebook, che non fa wait() sui figli)
            return False
        return abs(proc.create_time() - entry["pid_create_time"]) < 1.0
    except psutil.NoSuchProcess:
        return False


def _reconcile(reg: dict) -> None:
    """Marca 'crashed' i run 'running' il cui processo è morto senza avvisare (kill -9, riavvio...)."""
    for entry in reg["runs"].values():
        if entry["status"] == "running" and not _is_alive(entry):
            entry["status"] = "crashed"


def _first_fit_base(live_entries: list[dict], n_envs: int) -> int:
    """Il più piccolo base >= 0 con [base, base+n_envs) libero dagli intervalli dei run vivi."""
    base = 0
    for entry in sorted(live_entries, key=lambda e: e["instance_base"]):
        if base + n_envs <= entry["instance_base"]:
            break
        base = max(base, entry["instance_base"] + entry["n_envs"])
    return base


def launch(cfg: TrainConfig) -> RunHandle:
    """
    Avvia un training in background: assegna run_name/instance_base, congela la
    config in runs/<run>/config.json, spawna il worker e lo registra.
    """
    cfg = dataclasses.replace(cfg)  # non mutare l'oggetto del chiamante
    cfg.validate()

    with _locked_registry() as reg:
        _reconcile(reg)
        live = [e for e in reg["runs"].values() if e["status"] == "running"]

        run_id = reg["next_id"]
        if cfg.run_name is None:
            cfg.run_name = f"run_{run_id:03d}"
        if any(e["run_name"] == cfg.run_name for e in live):
            raise ValueError(f"run {cfg.run_name!r} è già in esecuzione")
        if cfg.run_name in reg["runs"]:
            print(f"attenzione: riuso il nome {cfg.run_name!r} di un run passato "
                  f"(checkpoints/tb_logs verranno condivisi)", file=sys.stderr)

        cfg.instance_base = _first_fit_base(live, cfg.n_envs)

        run_dir = RUNS_DIR / cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        log_path = run_dir / "train.log"
        cfg.to_json(config_path)

        with open(log_path, "w", buffering=1) as log:
            # start_new_session: sessione/process group propri -> sopravvive al
            # kernel Jupyter e stop() può fare killpg dell'intero albero
            proc = subprocess.Popen(
                [sys.executable, "-m", "smash_rl.train", "--config", str(config_path)],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                cwd=REPO_ROOT, start_new_session=True,
            )
        try:
            pid_create_time = psutil.Process(proc.pid).create_time()
        except psutil.NoSuchProcess:
            pid_create_time = 0.0  # morto subito: la riconciliazione lo marcherà crashed

        reg["runs"][cfg.run_name] = {
            "run_id": run_id,
            "run_name": cfg.run_name,
            "pid": proc.pid,
            "pid_create_time": pid_create_time,
            "instance_base": cfg.instance_base,
            "n_envs": cfg.n_envs,
            "config_path": str(config_path),
            "log_path": str(log_path),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "algo": cfg.algo,
            "total_steps": cfg.total_steps,
        }
        reg["next_id"] = run_id + 1

    print(f"{cfg.run_name}: pid {proc.pid}, istanze [{cfg.instance_base}, "
          f"{cfg.instance_base + cfg.n_envs}), porte {SLIPPI_BASE_PORT + cfg.instance_base}-"
          f"{SLIPPI_BASE_PORT + cfg.instance_base + cfg.n_envs - 1}, log {log_path}")
    return RunHandle(run_name=cfg.run_name, run_id=run_id, pid=proc.pid,
                     log_path=log_path, config_path=config_path)


def list_runs(all: bool = False) -> list[dict]:
    """Entry del registry (default: solo run vivi), ordinate per run_id."""
    with _locked_registry() as reg:
        _reconcile(reg)
        entries = [dict(e) for e in reg["runs"].values()
                   if all or e["status"] == "running"]
    return sorted(entries, key=lambda e: e["run_id"])


def status(run_name: str) -> dict:
    """Entry del run + liveness attuale + ultima riga di log."""
    with _locked_registry() as reg:
        _reconcile(reg)
        if run_name not in reg["runs"]:
            raise KeyError(f"run {run_name!r} non nel registry")
        entry = dict(reg["runs"][run_name])
    entry["alive"] = entry["status"] == "running"
    last = tail_log(run_name, n=1).strip()
    entry["last_log_line"] = last
    return entry


def tail_log(run_name: str, n: int = 30) -> str:
    """Ultime n righe del log del run (stringa vuota se il log non esiste ancora)."""
    log_path = RUNS_DIR / run_name / "train.log"
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def stop(run_name: str, timeout_s: float = 30.0) -> None:
    """
    Ferma un run in modo pulito: SIGTERM (il worker salva il modello e chiude i
    Dolphin nel suo finally), poi killpg SIGKILL se non muore entro timeout_s.
    """
    with _locked_registry() as reg:
        _reconcile(reg)
        if run_name not in reg["runs"]:
            raise KeyError(f"run {run_name!r} non nel registry")
        entry = reg["runs"][run_name]
        if entry["status"] != "running":
            print(f"{run_name}: già {entry['status']}, niente da fermare")
            return
        pid = entry["pid"]

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _is_alive(entry):
            break
        time.sleep(0.5)
    else:
        print(f"{run_name}: non risponde a SIGTERM, SIGKILL al process group", file=sys.stderr)
        try:
            os.killpg(pid, signal.SIGKILL)  # pgid == pid grazie a start_new_session
        except ProcessLookupError:
            pass

    mark_finished(run_name, "stopped")
    print(f"{run_name}: fermato")


def mark_finished(run_name: str, status: str) -> None:
    """Aggiorna lo stato finale di un run registrato; no-op per run in foreground non registrati."""
    with _locked_registry() as reg:
        entry = reg["runs"].get(run_name)
        if entry is not None:
            entry["status"] = status


def _format_table(entries: list[dict]) -> str:
    if not entries:
        return "(nessun run)"
    cols = ["run_name", "status", "pid", "algo", "instance_base", "n_envs",
            "total_steps", "started_at"]
    rows = [cols] + [[str(e[c]) for c in cols] for e in entries]
    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths))
                     for row in rows)


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gestione dei run di training")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").add_argument("--all", action="store_true",
                                        help="anche i run terminati")
    sub.add_parser("status").add_argument("run_name")
    p_stop = sub.add_parser("stop")
    p_stop.add_argument("run_name")
    p_stop.add_argument("--timeout", type=float, default=30.0)
    p_tail = sub.add_parser("tail")
    p_tail.add_argument("run_name")
    p_tail.add_argument("-n", type=int, default=30)
    sub.add_parser("launch").add_argument("config", help="path del TrainConfig in JSON")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        print(_format_table(list_runs(all=args.all)))
    elif args.cmd == "status":
        print(json.dumps(status(args.run_name), indent=2))
    elif args.cmd == "stop":
        stop(args.run_name, timeout_s=args.timeout)
    elif args.cmd == "tail":
        print(tail_log(args.run_name, n=args.n))
    elif args.cmd == "launch":
        launch(TrainConfig.from_json(args.config))


if __name__ == "__main__":
    main()
