import os

import melee
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from smash_rl.callbacks import EpisodeMetricsCallback
from smash_rl.factory import make_env
from smash_rl.session import kill_dolphin

"""
Lancia un addestramento DQN parallelo su Melee.

Uso: python train.py
I log dei worker finiscono in /tmp/melee_worker_N.log; i grafici in ./tb_logs/
(tensorboard --logdir tb_logs). I checkpoint in ./checkpoints/<RUN_NAME>/.
"""

# -- configurazione  --
RUN_NAME = "sixth_test"  # nome del run, usato per checkpoint e tensorboard
N_ENVS = 1
INSTANCE_BASE = 0        # offset di porte slippi/replay dir, utile per non collidere con altri training attivi
TOTAL_STEPS = 1_500_000
SKIP_KILL = False        # True = non uccidere i Dolphin esistenti (se c'è un altro training attivo)
CKPT_EVERY = 50_000      # salva i pesi ogni N timesteps totali (<= 0 = disabilitato)
PRETRAINED_MODEL_PATH = "dqn_melee_fifth_test.zip"   # path di un .zip per riprendere un addestramento

BOOT_STAGGER_S = 8.0     # sfasamento del boot tra worker, per non litigarsi le risorse all'avvio

ENV_KWARGS = dict(
    agent_char=melee.Character.FOX,
    opp_char=melee.Character.MARTH,
    opp_level=9,
    observation_function="pos_vel",
    action_function="a_only",
    reward_function="v1",
)

if __name__ == "__main__":
    if not SKIP_KILL:
        kill_dolphin()  # ATTENZIONE: uccide TUTTI i dolphin-emu, anche di altre run in corso

    venv = SubprocVecEnv(
        [make_env(INSTANCE_BASE + rank, seed=rank, save_name=RUN_NAME, boot_delay_s=rank * BOOT_STAGGER_S, **ENV_KWARGS)
         for rank in range(N_ENVS)],
        start_method="spawn",
    )
    print(f"SubprocVecEnv creato con {N_ENVS} istanze "
          f"(log dei worker: /tmp/melee_worker_N.log)", flush=True)

    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Carico il modello preaddestrato da {PRETRAINED_MODEL_PATH}")
        model = DQN.load(PRETRAINED_MODEL_PATH, env=venv, tensorboard_log="./tb_logs/")
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
            gamma=0.99,
            tensorboard_log="./tb_logs/",   # obbligatorio per avere i grafici
            verbose=1,
        )

    callbacks = [EpisodeMetricsCallback()]
    if CKPT_EVERY > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(CKPT_EVERY // N_ENVS, 1),   # save_freq conta gli step PER env
            save_path=f"./checkpoints/{RUN_NAME}/",
            name_prefix=f"dqn_melee_{RUN_NAME}",
        ))

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            tb_log_name=RUN_NAME,
            reset_num_timesteps=False,
            log_interval=1,
            callback=callbacks,
        )
    finally:
        # anche su ctrl-C o crash: salviamo i pesi e chiudiamo i worker (e i loro Dolphin)
        model.save(f"dqn_melee_{RUN_NAME}")
        venv.close()
