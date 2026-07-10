import argparse
import dataclasses
import json
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path

import melee
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv

from smash_rl.callbacks import EpisodeMetricsCallback
from smash_rl.factory import make_env
from smash_rl.session import kill_dolphin
from smash_rl.specs import ACT_SPECS, OBS_SPECS, REWARD_FNS

# algoritmi SB3 selezionabili
ALGOS = {"dqn": DQN, "ppo": PPO}

# chiavi di env_kwargs che vanno convertite da stringa a enum di melee
_ENUM_KEYS = {"agent_char": melee.Character, "opp_char": melee.Character,
              "stage": melee.Stage}


@dataclass
class TrainConfig:
    """
    Configurazione di un addestramento, serializzabile in JSON.

    Attributi:
        run_name: nome della run (serve per checkpoint, tensorboard, log)
        algo: algoritmo SB3 da usare ("dqn" o "ppo")
        policy: policy SB3 da usare ("MlpPolicy", "CnnPolicy", ...)
        algo_kwargs: parametri extra da passare al costruttore dell'algoritmo
        n_envs: numero di istanze parallele dell'ambiente (SubprocVecEnv)
        total_steps: numero totale di timesteps da addestrare
        ckpt_every: timesteps totali tra i checkpoint (<= 0 = disabilitato)
    """
    run_name: str | None = None      # se None, launch() genera run_NNN
    algo: str = "dqn"
    policy: str = "MlpPolicy"
    algo_kwargs: dict = field(default_factory=dict)
    n_envs: int = 4
    total_steps: int = 3_000_000
    ckpt_every: int = 50_000         # timesteps totali tra i checkpoint (<= 0 = disabilitato)
    seed: int = 0
    boot_stagger_s: float = 8.0      # sfasamento del boot tra worker
    instance_base: int | None = None  # primo instance_id (porta 51441+i); lo assegna launch()
    pretrained_path: str | None = None  # .zip da cui riprendere l'addestramento
    reset_exploration: dict | None = None  # solo DQN, con pretrained_path: {"initial_eps", "final_eps", "fraction"}
    kill_all_dolphins: bool = False  # opt-in: pkill globale di dolphin-emu (NO con altri run attivi)
    env_kwargs: dict = field(default_factory=lambda: dict(
        agent_char="FOX",
        opp_char="MARTH",
        opp_level=7,
        observation_function="pos_vel",
        action_function="a_only",
        reward_function="v1",
    ))

    def validate(self) -> None:
        if self.algo not in ALGOS:
            raise ValueError(f"algo {self.algo!r} sconosciuto (disponibili: {sorted(ALGOS)})")
        for key, registry in [("observation_function", OBS_SPECS),
                              ("action_function", ACT_SPECS),
                              ("reward_function", REWARD_FNS)]:
            name = self.env_kwargs.get(key)
            if name is not None and name not in registry:
                raise ValueError(f"{key} {name!r} sconosciuta (disponibili: {sorted(registry)})")
        for key, enum in _ENUM_KEYS.items():
            name = self.env_kwargs.get(key)
            if isinstance(name, str) and name not in enum.__members__:
                raise ValueError(f"{key} {name!r} non è un membro di melee.{enum.__name__}")
        if self.reset_exploration is not None:
            if self.algo != "dqn":
                raise ValueError("reset_exploration ha senso solo con algo='dqn'")
            missing = {"initial_eps", "final_eps", "fraction"} - self.reset_exploration.keys()
            if missing:
                raise ValueError(f"reset_exploration incompleto, mancano: {sorted(missing)}")
        if self.n_envs < 1:
            raise ValueError("n_envs deve essere >= 1")

    def to_json(self, path) -> None:
        """
        Salva un TrainConfig in JSON
        
        Args:
            path: percorso del file JSON in cui salvare la config
        """

        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")

    @classmethod
    def from_json(cls, path) -> "TrainConfig":
        """
        Carica un TrainConfig da JSON
        
        Args:
            path: percorso del file JSON da cui caricare la config
        """

        data = json.loads(Path(path).read_text())
        unknown = data.keys() - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"campi sconosciuti in {path}: {sorted(unknown)}")
        return cls(**data)


def resolve_env_kwargs(cfg: TrainConfig) -> dict:
    """env_kwargs con i nomi di personaggi/stage convertiti negli enum di melee."""
    kwargs = dict(cfg.env_kwargs)
    for key, enum in _ENUM_KEYS.items():
        if isinstance(kwargs.get(key), str):
            kwargs[key] = enum[kwargs[key]]
    return kwargs


def run(cfg: TrainConfig) -> None:
    """Esegue un addestramento completo (bloccante) secondo la config."""
    cfg.validate()
    if cfg.run_name is None:
        raise ValueError("run_name mancante (assegnalo o passa da runs.launch)")
    if cfg.instance_base is None:
        raise ValueError("instance_base mancante: serve per non collidere con altri run "
                         "(0 se non ce ne sono; runs.launch lo assegna da solo)")

    if cfg.kill_all_dolphins:
        kill_dolphin()  # ATTENZIONE: uccide TUTTI i dolphin-emu, anche di altre run in corso

    env_kwargs = resolve_env_kwargs(cfg)
    venv = SubprocVecEnv(
        [make_env(cfg.instance_base + rank, seed=cfg.seed + rank, save_name=cfg.run_name,
                  boot_delay_s=rank * cfg.boot_stagger_s, **env_kwargs)
         for rank in range(cfg.n_envs)],
        start_method="spawn",
    )
    print(f"SubprocVecEnv creato con {cfg.n_envs} istanze, base {cfg.instance_base} "
          f"(log dei worker: /tmp/melee_worker_N.log)", flush=True)

    algo_cls = ALGOS[cfg.algo]
    if cfg.pretrained_path:
        print(f"Carico il modello preaddestrato da {cfg.pretrained_path}", flush=True)
        model = algo_cls.load(cfg.pretrained_path, env=venv, tensorboard_log="./tb_logs/")
        if cfg.reset_exploration:
            # riparte con una policy già sensata: si abbassa l'esplorazione iniziale
            model.exploration_initial_eps = cfg.reset_exploration["initial_eps"]
            model.exploration_final_eps = cfg.reset_exploration["final_eps"]
            model.exploration_fraction = cfg.reset_exploration["fraction"]
            model.exploration_schedule = LinearSchedule(
                model.exploration_initial_eps,
                model.exploration_final_eps,
                model.exploration_fraction,
            )
    else:
        model = algo_cls(cfg.policy, venv, tensorboard_log="./tb_logs/", verbose=1,
                         **cfg.algo_kwargs)

    ckpt_dir = Path("checkpoints") / cfg.run_name
    callbacks = [EpisodeMetricsCallback()]
    if cfg.ckpt_every > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(cfg.ckpt_every // cfg.n_envs, 1),  # save_freq conta gli step PER env
            save_path=str(ckpt_dir),
            name_prefix=f"{cfg.algo}_melee_{cfg.run_name}",
        ))

    try:
        model.learn(
            total_timesteps=cfg.total_steps,
            tb_log_name=cfg.run_name,
            reset_num_timesteps=True,
            log_interval=1,
            callback=callbacks,
        )
    finally:
        # anche su stop/ctrl-C/crash: salviamo i pesi e chiudiamo i worker (e i loro Dolphin)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save(ckpt_dir / "final")
        venv.close()


def _parse_args(argv=None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Addestramento Melee (foreground)")
    parser.add_argument("--config", required=True, help="path del TrainConfig in JSON")
    parser.add_argument("--run-name")
    parser.add_argument("--algo", choices=sorted(ALGOS))
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--instance-base", type=int)
    parser.add_argument("--pretrained", dest="pretrained_path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ckpt-every", type=int)
    args = parser.parse_args(argv)

    cfg = TrainConfig.from_json(args.config)
    for name in ("run_name", "algo", "n_envs", "total_steps", "instance_base",
                 "pretrained_path", "seed", "ckpt_every"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    return cfg


def main(argv=None) -> None:
    cfg = _parse_args(argv)
    if cfg.run_name is None:
        raise SystemExit("serve --run-name (o run_name nel config)")

    # SIGTERM (da runs.stop o dal sistema) deve svolgere il finally di run():
    # salvataggio del modello e chiusura dei worker/Dolphin.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

    status = "crashed"
    try:
        run(cfg)
        status = "finished"
    except (SystemExit, KeyboardInterrupt):
        status = "stopped"
        raise
    finally:
        from smash_rl import runs
        runs.mark_finished(cfg.run_name, status)


if __name__ == "__main__":
    main()
