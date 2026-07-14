"""
Rollout "live" di un modello addestrato contro la CPU, con salvataggio dei
record per-decisione: observation, Q-values, azione giocata e attivazioni dei
layer nascosti.

A differenza di video/diagnostics.extract_frame_records (che rilegge un .slp e
RICALCOLA la greedy senza frame_skip), qui il modello gioca davvero dentro
MeleeEnv: i record sono allineati alle azioni effettivamente eseguite, una per
decisione (cioè ogni frame_skip frame di gioco).

Ogni forward su q_net produce, in un colpo solo, i Q-values e (via hook) le
attivazioni; argmax(q) coincide con model.predict(deterministic=True). L'azione
si passa a env.step, che ritorna reward e info (stock/percent) del passo.

Uso da CLI:
    python -m smash_rl.rollout weights/final.zip --episodes 5 --out rollouts/final
    python -m smash_rl.rollout weights/final.zip --opp-level 7 --opp-char MARTH \
        --obs full_obs --act a_b --reward v4 --frame-skip 3

Output: una cartella con un <NNN>.npz per episodio + meta.json (spec, avversario,
nomi delle feature, etichette delle azioni, dimensioni dei layer, esito).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import melee
import numpy as np
import torch

from smash_rl.environment import MeleeEnv
from smash_rl.specs.actions import ACT_LABELS, ACT_LAYOUTS, ACT_SPECS, button_label
from smash_rl.specs.observations import OBS_FEATURE_NAMES, OBS_SPECS
from smash_rl.video.diagnostics import _hook_activations, load_model


def _validate_model(model, obs_name: str, act_name: str):
    """Il checkpoint deve combaciare con le spec richieste (come play.py)."""
    obs_space, _ = OBS_SPECS[obs_name]
    act_space, decode_act = ACT_SPECS[act_name]
    if model.observation_space.shape != obs_space.shape:
        raise ValueError(f"il checkpoint si aspetta obs {model.observation_space.shape}, "
                         f"ma '{obs_name}' produce {obs_space.shape}: usa obs_name giusto")
    if model.action_space.n != act_space.n:
        raise ValueError(f"il checkpoint ha {model.action_space.n} azioni, "
                         f"ma '{act_name}' ne definisce {act_space.n}: usa act_name giusto")
    return decode_act


def run_rollout(model_path, out_dir, *,
                n_episodes: int = 5,
                obs_name: str = "full_obs",
                act_name: str = "a_b",
                reward_name: str = "v4",
                agent_char: melee.Character = melee.Character.FOX,
                opp_char: melee.Character = melee.Character.MARTH,
                opp_level: int = 7,
                stage: melee.Stage = melee.Stage.FINAL_DESTINATION,
                frame_skip: int = 3,
                capture_activations: bool = True,
                max_reset_attempts: int = 5,
                reset_timeout_s: float = 120.0,
                advance_timeout_s: float = 90.0,
                config=None) -> Path:
    """
    Fa giocare `model_path` contro la CPU per `n_episodes` partite e salva i
    record per-decisione in `out_dir` (un .npz per episodio + meta.json).
    Ritorna la cartella di output.

    `config` (MeleeConfig) è opzionale: passalo per fissare porta Slippi/home
    Dolphin uniche (usato dal launcher parallelo). Se None si usa il .env
    (istanza 0). Questa funzione fa girare UNA sola istanza di Dolphin.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    decode_act = _validate_model(model, obs_name, act_name)

    q_net = model.policy.q_net
    captured, handles = ([], [])
    if capture_activations:
        captured, handles = _hook_activations(q_net.q_net)

    env = MeleeEnv(
        config=config,
        agent_char=agent_char, opp_char=opp_char, opp_level=opp_level, stage=stage,
        observation_function=obs_name, action_function=act_name,
        reward_function=reward_name, frame_skip=frame_skip,
        max_reset_attempts=max_reset_attempts, reset_timeout_s=reset_timeout_s,
        advance_timeout_s=advance_timeout_s,
    )
    replay_dir = Path(env.session.config.replay_dir)   # dove Dolphin scrive le .slp

    episodes_meta = []
    try:
        with torch.no_grad():
            for ep in range(n_episodes):
                slp_before = _list_slp(replay_dir)   # snapshot pre-partita per il diff
                obs, _info = env.reset()
                rec = _new_episode_buffers(len(captured) if capture_activations else 0)
                done = False
                last_info = {}
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    q = q_net(obs_t).numpy().squeeze(0)          # riempie anche `captured`
                    action = int(q.argmax())
                    (sx, sy), button = decode_act(action)

                    next_obs, reward, term, trunc, info = env.step(action)

                    _append_step(rec, obs, q, action, sx, sy, button, reward, info,
                                 captured if capture_activations else None)

                    obs, done, last_info = next_obs, (term or trunc), info

                slp_name = _match_slp(replay_dir, slp_before)   # .slp comparsa in questo episodio
                out_path = out_dir / f"{ep:03d}.npz"
                _save_episode(out_path, rec, slp_name)
                won = int(last_info.get("P2_stocks", 1)) <= 0
                episodes_meta.append({
                    "file": out_path.name, "steps": len(rec["action"]),
                    "won": bool(won), "slp_file": slp_name,
                    "final_info": {k: int(v) for k, v in last_info.items()
                                   if k.startswith("P")},
                })
                print(f"[rollout] episodio {ep + 1}/{n_episodes}: "
                      f"{len(rec['action'])} decisioni, {'VINTO' if won else 'perso'} "
                      f"-> {out_path.name} (slp: {slp_name or 'n/d'})", flush=True)
    finally:
        for h in handles:
            h.remove()
        env.close()

    meta = {
        "model": str(model_path),
        "obs_name": obs_name, "act_name": act_name, "reward_name": reward_name,
        "agent_char": agent_char.name, "opp_char": opp_char.name,
        "opp_level": opp_level, "stage": stage.name, "frame_skip": frame_skip,
        "replay_dir": str(replay_dir),
        "obs_feature_names": OBS_FEATURE_NAMES.get(obs_name),
        "action_labels": ACT_LABELS.get(act_name),
        "action_layout": ACT_LAYOUTS.get(act_name),
        "n_activation_layers": len(captured) if capture_activations else 0,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes": episodes_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[rollout] fatto: {n_episodes} episodi in {out_dir} (meta.json scritto)", flush=True)
    return out_dir


def _list_slp(replay_dir: Path) -> set[str]:
    """I .slp attualmente presenti in replay_dir (ricorsivo). Set vuoto se manca."""
    if not replay_dir.exists():
        return set()
    return {str(p) for p in replay_dir.rglob("*.slp")}


def _match_slp(replay_dir: Path, before: set[str]) -> str:
    """
    Nome del .slp comparso *dopo* lo snapshot `before`: è la replay dell'episodio
    appena giocato. Se ne compaiono più d'uno (es. un match abortito durante il
    reset), prende il più grande = la partita vera. Ritorna il path relativo a
    replay_dir (o "" se non trovato / save_replays disattivo).
    """
    new = _list_slp(replay_dir) - before
    if not new:
        return ""
    best = max(new, key=lambda p: Path(p).stat().st_size if Path(p).exists() else 0)
    return str(Path(best).relative_to(replay_dir))


def _new_episode_buffers(n_layers: int) -> dict:
    buf = {k: [] for k in ("obs", "q_values", "action", "stick", "button",
                           "reward", "P1_stocks", "P2_stocks", "P1_percent", "P2_percent")}
    buf["_act"] = [[] for _ in range(n_layers)]   # una lista per layer nascosto
    return buf


def _append_step(rec, obs, q, action, sx, sy, button, reward, info, captured):
    rec["obs"].append(np.asarray(obs, np.float32))
    rec["q_values"].append(q.astype(np.float32))
    rec["action"].append(action)
    rec["stick"].append((sx, sy))
    rec["button"].append(button_label(button))
    rec["reward"].append(float(reward))
    for k in ("P1_stocks", "P2_stocks", "P1_percent", "P2_percent"):
        rec[k].append(info.get(k, 0))
    if captured is not None:
        for i, a in enumerate(captured):
            rec["_act"][i].append(np.asarray(a, np.float32))


def _save_episode(path: Path, rec: dict, slp_name: str = "") -> None:
    arrays = {
        "slp_file": np.asarray(slp_name),                # () str: replay Slippi associata
        "obs": np.stack(rec["obs"]),                     # (T, obs_dim)
        "q_values": np.stack(rec["q_values"]),           # (T, n_actions)
        "action": np.asarray(rec["action"], np.int64),   # (T,)
        "stick": np.asarray(rec["stick"], np.float32),   # (T, 2)
        "button": np.asarray(rec["button"], dtype="U4"), # (T,) es. "A","B","-"
        "reward": np.asarray(rec["reward"], np.float32),
        "P1_stocks": np.asarray(rec["P1_stocks"], np.float32),
        "P2_stocks": np.asarray(rec["P2_stocks"], np.float32),
        "P1_percent": np.asarray(rec["P1_percent"], np.float32),
        "P2_percent": np.asarray(rec["P2_percent"], np.float32),
    }
    for i, layer in enumerate(rec["_act"]):              # activations_l0, l1, ... (T, units)
        arrays[f"activations_l{i}"] = np.stack(layer) if layer else np.empty((0,), np.float32)
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Launcher parallelo: N istanze di Dolphin in parallelo, una per processo.
# ---------------------------------------------------------------------------

def _split_episodes(n_episodes: int, n_workers: int) -> list[int]:
    """Distribuisce n_episodes il più equamente possibile su n_workers (>0 solo)."""
    base, rem = divmod(n_episodes, n_workers)
    counts = [base + (1 if i < rem else 0) for i in range(n_workers)]
    return [c for c in counts if c > 0]


def _rollout_worker(instance_id, model_path, worker_dir, n_episodes,
                    save_name, log_path, kwargs):
    """
    Corpo di un processo worker: config per-istanza (porta/home uniche), log su
    file, poi run_rollout nella propria sottocartella. Eseguito via 'spawn',
    quindi tutto ciò che riceve dev'essere picklabile.
    """
    import ctypes
    import os
    import signal
    import sys
    import traceback

    from smash_rl.session import MeleeConfig

    try:   # se il padre muore, ricevi SIGTERM invece di restare appeso (come factory.make_env)
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:
        pass

    log = open(log_path, "w", buffering=1)
    sys.stdout, sys.stderr = log, log
    try:
        cfg = MeleeConfig.for_instance(instance_id, save_name=save_name)
        print(f"[worker {instance_id}] pid={os.getpid()} porta={cfg.slippi_port} "
              f"episodi={n_episodes}", flush=True)
        run_rollout(model_path, worker_dir, n_episodes=n_episodes, config=cfg, **kwargs)
    except Exception:
        traceback.print_exc()
        raise


def run_rollouts(model_path, out_dir=None, *, n_episodes: int = 10, n_workers: int = 4,
                 save_name: str = "rollout", instance_base: int = 0,
                 log_dir="/tmp/melee_rollout_logs", **kwargs) -> Path:
    """
    Launcher comodo (pensato per il notebook): fa girare `n_episodes` partite del
    modello contro la CPU su `n_workers` istanze di Dolphin in parallelo e salva
    i record. Ritorna la cartella di output, pronta per `load_rollout`.

    Ogni worker scrive nella propria sottocartella `worker_<i>/`; `load_rollout`
    le unisce trasparentemente. I `kwargs` (obs_name, act_name, reward_name,
    agent_char, opp_char, opp_level, stage, frame_skip, capture_activations)
    passano dritti a `run_rollout`.

    instance_base: primo instance_id (porta Slippi 51441+id); alzalo se hai
    altri run/istanze attive per non collidere sulle porte.
    """
    out_dir = Path(out_dir) if out_dir else (Path("rollouts") / Path(model_path).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_workers <= 1:   # percorso semplice, in-process (niente multiprocessing)
        return run_rollout(model_path, out_dir, n_episodes=n_episodes, **kwargs)

    per_worker = _split_episodes(n_episodes, n_workers)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")   # come SubprocVecEnv nel training
    procs = []
    for i, count in enumerate(per_worker):
        worker_dir = out_dir / f"worker_{i}"
        log_path = str(log_dir / f"rollout_worker_{i}.log")
        p = ctx.Process(target=_rollout_worker,
                        args=(instance_base + i, str(model_path), str(worker_dir),
                              count, save_name, log_path, kwargs))
        p.start()
        procs.append((i, p))

    print(f"[rollouts] {len(procs)} worker avviati ({n_episodes} episodi totali); "
          f"log in {log_dir}/rollout_worker_*.log", flush=True)
    failed = []
    for i, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append((i, p.exitcode))

    if failed:
        print(f"[rollouts] ATTENZIONE: worker falliti {failed} "
              f"(vedi i log in {log_dir})", flush=True)
    print(f"[rollouts] fatto -> {out_dir}", flush=True)
    return out_dir


# ---------------------------------------------------------------------------
# Loader: ricarica i record salvati per l'analisi nel notebook.
# ---------------------------------------------------------------------------

class Episode:
    """Un episodio caricato: array per-passo + attivazioni per layer."""

    _SCALAR = ("action", "reward", "P1_stocks", "P2_stocks",
               "P1_percent", "P2_percent")

    def __init__(self, data: dict, file: str):
        self._d = data
        self.file = file
        self.slp_file = str(data["slp_file"]) if "slp_file" in data else ""  # replay Slippi
        self.obs = data["obs"]                 # (T, obs_dim)
        self.q_values = data["q_values"]       # (T, n_actions)
        self.action = data["action"]           # (T,)
        self.stick = data["stick"]             # (T, 2)
        self.button = data["button"]           # (T,) str
        self.reward = data["reward"]
        self.P1_stocks = data["P1_stocks"]
        self.P2_stocks = data["P2_stocks"]
        self.P1_percent = data["P1_percent"]
        self.P2_percent = data["P2_percent"]
        # activations_l0, l1, ... nell'ordine dei layer nascosti
        self.activations = [data[k] for k in sorted(data)
                            if k.startswith("activations_l")]

    @property
    def n_steps(self) -> int:
        return len(self.action)

    @property
    def won(self) -> bool:
        return bool(self.P2_stocks[-1] <= 0) if self.n_steps else False

    def __len__(self) -> int:
        return self.n_steps

    def __repr__(self) -> str:
        return (f"Episode({self.file}, steps={self.n_steps}, "
                f"{'won' if self.won else 'lost'})")


class Rollout:
    """
    Insieme di episodi caricati da una cartella di rollout, con i metadati
    (spec, nomi delle feature, etichette delle azioni). Iterabile e indicizzabile.
    """

    def __init__(self, episodes: list[Episode], meta: dict, path: Path):
        self.episodes = episodes
        self.meta = meta
        self.path = path

    # comodità
    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i) -> Episode:
        return self.episodes[i]

    def __iter__(self):
        return iter(self.episodes)

    @property
    def obs_feature_names(self):
        return self.meta.get("obs_feature_names")

    @property
    def action_labels(self):
        return self.meta.get("action_labels")

    @property
    def win_rate(self) -> float:
        return float(np.mean([e.won for e in self.episodes])) if self.episodes else 0.0

    def concat(self) -> dict:
        """
        Concatena tutti i passi di tutti gli episodi in array unici, con un
        `episode_id` (T_tot,) che indica l'episodio di provenienza. Le attivazioni
        diventano una lista di array (T_tot, units), una per layer.
        """
        if not self.episodes:
            return {}
        out = {"episode_id": np.concatenate(
            [np.full(e.n_steps, i, np.int64) for i, e in enumerate(self.episodes)])}
        for key in ("obs", "q_values", "action", "stick", "button", "reward",
                    "P1_stocks", "P2_stocks", "P1_percent", "P2_percent"):
            out[key] = np.concatenate([getattr(e, key) for e in self.episodes])
        n_layers = len(self.episodes[0].activations)
        out["activations"] = [
            np.concatenate([e.activations[l] for e in self.episodes])
            for l in range(n_layers)
        ]
        return out

    def to_dataframe(self):
        """
        DataFrame pandas dei campi scalari per-passo (episode_id, step, action,
        button, reward, stock/percent) + una colonna q_<label> per ogni azione.
        Le obs e le attivazioni restano fuori (usa `concat`). Richiede pandas.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "to_dataframe richiede pandas (opzionale): pip install pandas. "
                "In alternativa usa .concat() per gli array numpy."
            ) from e

        c = self.concat()
        if not c:
            return pd.DataFrame()
        step = np.concatenate([np.arange(e.n_steps) for e in self.episodes])
        slp = np.concatenate([np.full(e.n_steps, e.slp_file) for e in self.episodes])
        df = pd.DataFrame({
            "episode_id": c["episode_id"], "step": step, "slp_file": slp,
            "action": c["action"], "button": c["button"],
            "stick_x": c["stick"][:, 0], "stick_y": c["stick"][:, 1],
            "reward": c["reward"],
            "P1_stocks": c["P1_stocks"], "P2_stocks": c["P2_stocks"],
            "P1_percent": c["P1_percent"], "P2_percent": c["P2_percent"],
        })
        labels = self.action_labels or [f"a{i}" for i in range(c["q_values"].shape[1])]
        for i, lab in enumerate(labels):
            df[f"q_{lab}"] = c["q_values"][:, i]
        return df

    def __repr__(self) -> str:
        return (f"Rollout({self.path}, episodes={len(self.episodes)}, "
                f"win_rate={self.win_rate:.2f})")


def load_rollout(out_dir, *, load_activations: bool = True) -> Rollout:
    """
    Ricarica una cartella prodotta da `run_rollout`/`run_rollouts` (cerca i .npz
    ricorsivamente, quindi copre anche le sottocartelle worker_*/). Ritorna un
    `Rollout` con gli episodi ordinati e i metadati. Con `load_activations=False`
    salta le attivazioni (file grandi) per un caricamento più leggero.
    """
    out_dir = Path(out_dir)
    npz_files = sorted(out_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"nessun .npz in {out_dir}")

    episodes = []
    for f in npz_files:
        with np.load(f, allow_pickle=False) as d:
            data = {k: d[k] for k in d.files
                    if load_activations or not k.startswith("activations_l")}
        episodes.append(Episode(data, file=str(f.relative_to(out_dir))))

    # metadati: uniamo gli episodes di tutti i meta.json (la spec è condivisa)
    meta: dict = {}
    metas = sorted(out_dir.rglob("meta.json"))
    if metas:
        meta = json.loads(metas[0].read_text())
        merged = []
        for m in metas:
            merged.extend(json.loads(m.read_text()).get("episodes", []))
        meta["episodes"] = merged
    return Rollout(episodes, meta, out_dir)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="checkpoint .zip dell'agente")
    p.add_argument("--out", type=Path, default=None,
                   help="cartella di output (default: rollouts/<nome-modello>)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--workers", type=int, default=1,
                   help="istanze di Dolphin in parallelo (default 1)")
    p.add_argument("--instance-base", type=int, default=0,
                   help="primo instance_id/porta (alzalo se hai altri run attivi)")
    p.add_argument("--obs", default="full_obs", choices=sorted(OBS_SPECS))
    p.add_argument("--act", default="a_b", choices=sorted(ACT_SPECS))
    p.add_argument("--reward", default="v4")
    p.add_argument("--agent-char", default="FOX")
    p.add_argument("--opp-char", default="MARTH")
    p.add_argument("--opp-level", type=int, default=7)
    p.add_argument("--stage", default="FINAL_DESTINATION")
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--no-activations", action="store_true",
                   help="non salvare le attivazioni dei layer nascosti")
    args = p.parse_args(argv)

    out = args.out or (Path("rollouts") / Path(args.model).stem)
    run_rollouts(
        args.model, out, n_episodes=args.episodes, n_workers=args.workers,
        instance_base=args.instance_base,
        obs_name=args.obs, act_name=args.act, reward_name=args.reward,
        agent_char=melee.Character[args.agent_char.upper()],
        opp_char=melee.Character[args.opp_char.upper()],
        opp_level=args.opp_level, stage=melee.Stage[args.stage.upper()],
        frame_skip=args.frame_skip, capture_activations=not args.no_activations,
    )


if __name__ == "__main__":
    main()
