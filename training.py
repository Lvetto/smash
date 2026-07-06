from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import DQN
from melee_env import MeleeEnv, MeleeConfig
import sys, os
import multiprocessing as mp
from lib import kill_dolphin

def make_env(rank, seed=0):
    def _init():
        mp.current_process()._config["daemon"] = False  # workaround per evitare errori di multiprocessing con spawn
        log = open(f"/tmp/melee_worker_{rank}.log", "w", buffering=1)
        sys.stdout, sys.stderr = log, log
        cfg = MeleeConfig.for_instance(rank)
        env = MeleeEnv(config=cfg)
        env = Monitor(env)
        pid = os.getpid()
        print(f"Worker {rank} started with PID {pid}")
        return env
    return _init

PRETRAINED_MODEL_PATH = "dqn_melee_phase1.zip"

if __name__ == "__main__":
    kill_dolphin()  # Kill any existing Dolphin processes

    n_envs = 1
    venv = SubprocVecEnv(
        [make_env(i) for i in range(n_envs)],
        start_method="spawn",                  # niente fork da processo multithread
    )

    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Loading pretrained model from {PRETRAINED_MODEL_PATH}")
        model = DQN.load(
            PRETRAINED_MODEL_PATH,
            env=venv,
            tensorboard_log="./tb_logs/",
        )
        # Se hai salvato anche il replay buffer (model.save_replay_buffer(...)),
        # ricaricalo qui per non ripartire da un buffer vuoto:
        # model.load_replay_buffer("dqn_melee_phase1_buffer.pkl")
    else:
        model = DQN(
        "MlpPolicy", venv,
        buffer_size=500_000,
        learning_starts=5_000,
        train_freq=1,
        gradient_steps=1,
        batch_size=64,
        target_update_interval=1_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        tensorboard_log="./tb_logs/",       # obbligatorio per avere i grafici
        verbose=1,
        gamma=0.999,
        )

    model.learn(
    total_timesteps=500_000 * 6,    # 500k passi = circa 1 ora di addestramento
    tb_log_name="phase2_stationary",
    reset_num_timesteps=False,      # continua il conteggio invece di azzerarlo
    )

    model.save("dqn_melee_phase2_stationary")
    venv.close()
