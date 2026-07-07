from stable_baselines3.common.monitor import Monitor
from smash_rl.environment import MeleeEnv
import multiprocessing as mp
import ctypes, signal
import sys, os, traceback

from smash_rl.session import MeleeConfig

def make_env(instance_id, seed, save_name, worker_mode=True, **kwargs):
    """
    Crea un'istanza dell'ambiente MeleeEnv con le specifiche fornite.

    Args:
        instance_id (int): L'ID dell'istanza dell'ambiente.
        seed (int): Il seme per la generazione casuale.
        save_name (str): Il nome del run (usato per la replay_dir).
        worker_mode (bool): True per l'uso in un worker di SubprocVecEnv, False per l'uso in un processo principale.
        **kwargs: Altri argomenti da passare al costruttore di MeleeEnv.

    Returns:
        function: Una funzione che inizializza e restituisce l'ambiente MeleeEnv.
    """
    def _init():

        mp.current_process()._config["daemon"] = False  # per SlippStream

        if worker_mode:
            try:    # workaround per recuperare il comportamento alla morte di un daemon: se il processo padre muore, anche i figli ricevono SIGTERM. Senza questo, i worker restano appesi se il main crasha.
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                PR_SET_PDEATHSIG = 1
                libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
            except Exception:
                pass

            log = open(f"/tmp/melee_worker_{instance_id}.log", "w", buffering=1)   # apriamo un file di log per istanza
            sys.stdout, sys.stderr = log, log                                      # reindirizziamo stdout e stderr al file di log

        try:
            cfg = MeleeConfig.for_instance(instance_id, save_name=save_name)  # otteniamo la configurazione specifica per l'istanza
            print(f"Worker {instance_id} started with PID {os.getpid()} (porta slippi {cfg.slippi_port})", flush=True)

            env = MeleeEnv(config=cfg, **kwargs)  # creiamo l'ambiente MeleeEnv con la configurazione e gli argomenti forniti

            if worker_mode:
                def _terminate(signum, frame):
                    # il main è morto: uccidi il tuo Dolphin e il display, poi esci subito
                    print(f"Worker {instance_id}: SIGTERM (main morto?), chiudo Dolphin ed esco", flush=True)
                    try:
                        env.close()
                    finally:
                        os._exit(1)
                signal.signal(signal.SIGTERM, _terminate)

            env = Monitor(env, info_keywords=("P1_stocks", "P2_stocks", "P1_percent", "P2_percent"))  # aggiunge info al log
            return env
        except Exception:
            # senza questo, un errore in fase di creazione muore silenziosamente
            # nel log e il processo principale resta appeso senza spiegazione (successo più volte)
            traceback.print_exc()
            raise

    return _init
