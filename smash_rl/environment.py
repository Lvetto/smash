import gymnasium as gym
from gymnasium import spaces
import numpy as np
import melee
from melee.slippstream import EnetDisconnected
from smash_rl.session import MeleeSession, MeleeConfig, PlayerSpec
from smash_rl.specs.actions import ACT_SPECS
from smash_rl.specs.observations import OBS_SPECS
from smash_rl.specs.rewards import REWARD_FNS
from smash_rl.specs.context import Ctx
from smash_rl.session import run_with_watchdog, WatchdogTimeout
import time

# Eccezioni da cui si recupera con un reboot pulito dell'ambiente. Tutti gli altri errori sono considerati fatali e fanno crashare il worker:
# - Timeout: watchdog scattato
# - TimeoutError: fase di boot/menu oltre il tempo massimo
# - RuntimeError: connect fallito o Dolphin morto al boot
# - EnetDisconnected: Dolphin morto/scollegato a metà partita (succede)
# - BrokenPipeError: scrittura sul pipe di un Dolphin morto
RECOVERABLE_ERRORS = (WatchdogTimeout, TimeoutError, RuntimeError, EnetDisconnected, BrokenPipeError)

# -- definizione dell'ambiente Gymnasium per il training --
class MeleeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None,
                agent_char=melee.Character.FOX,
                opp_char=melee.Character.MARTH,
                opp_level=2,
                stage=melee.Stage.FINAL_DESTINATION,
                boot_delay_s=None,
                debug=False,
                observation_function="full_v1",
                action_function="full",
                reward_function="v1",
                victory_bonus=2.0,
                frame_skip=3,
                max_reset_attempts=3,
                reset_timeout_s=90.0,
                advance_timeout_s=60.0,):
        
        self.spec_names = dict(obs=observation_function , act=action_function, reward=reward_function)  # per il logging
        self.observation_space, self._build  = OBS_SPECS[observation_function]
        self.action_space,      self._decode = ACT_SPECS[action_function]
        self._reward = REWARD_FNS[reward_function]

        self._last_obs = np.zeros(self.observation_space.shape, np.float32)  # unica fonte di verità
        self._prev_gs = None

        self.stage = stage
        self.agent_idx, self.opp_idx = 0, 1   # indici dei controller

        # porta 1 = agente (cpu_level=0 -> controller reale), porta 2 = CPU
        players = [PlayerSpec(agent_char, cpu_level=0),
                   PlayerSpec(opp_char, cpu_level=opp_level)]

        self.session = MeleeSession(config=config or MeleeConfig.from_env(), players=players)
        self.ctx = Ctx(agent_port=1, opp_port=2, session=self.session)  # dati necessari per le observation/reward che leggono le properties della session

        self._steps = 0

        self.debug = debug   # True = stampa un log per ogni frame (log ENORMI, solo per debug)

        self.win_bonus = victory_bonus

        # Parametri per il frame skip e il numero massimo di passi
        self.frame_skip = frame_skip
        self.max_steps = int(7.5 * 60 * 60 / self.frame_skip)  # 7.5 min * 60fps / skip = numero di decisioni

        # Per evitare blocchi in caso di processi appesi (configurabili: sotto forte
        # pressione di memoria/swap la navigazione menu rallenta e conviene alzarli)
        self.max_reset_attempts = max_reset_attempts
        self.reset_timeout_s = reset_timeout_s
        self.advance_timeout_s = advance_timeout_s   # timeout navigazione menu, DEVE essere < reset_timeout_s
        assert self.advance_timeout_s < self.reset_timeout_s, \
            "advance_timeout_s deve essere < reset_timeout_s (il watchdog di reset lo contiene)"

        # per multiprocessing
        self._booted_once = False
        self.boot_stagger_s = 8.0   # secondi di sfasamento per istanza (usato solo se boot_delay_s non è passato)
        self.boot_delay_s = (boot_delay_s if boot_delay_s is not None       # non le facciamo partire allo stesso tempo per non litigarsi risorse
                             else self.session.config.instance_id * self.boot_stagger_s)
        self.step_timeout_s = 12.0

    def _log(self, msg):
        print(f"[env inst={self.session.config.instance_id}] {msg}", flush=True)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if not self._booted_once:   # attesa al primo avvio per evitare contesa di risorse tra più istanze
            time.sleep(self.boot_delay_s)
            self._booted_once = True

        last_err = None
        for attempt in range(self.max_reset_attempts):        # es. 3
            try:
                self._log(f"reset: tentativo {attempt + 1}/{self.max_reset_attempts}")
                obs_info = run_with_watchdog(
                    self._do_reset,
                    timeout_s=self.reset_timeout_s,           # 90.0 (boot + menu con margine)
                    on_timeout=self.session.force_kill_dolphin,
                )
                self._log("reset: in game")
                return obs_info
            # Se abbiamo un errore rimediabile, chiudiamo tutto e ricostruiamo l'ambiente da zero. Il VecEnv chiama reset() di nuovo, che fa un nuovo tentativo.
            except RECOVERABLE_ERRORS as e:
                last_err = e
                self._log(f"reset: tentativo fallito ({type(e).__name__}: {e}), retry")
                self.session.close()   # già gestito in maniera pulita dalla session
                time.sleep(2.0)        # margine per rilascio porta/risorse prima del re-boot
        raise RuntimeError(
            f"reset fallito dopo {self.max_reset_attempts} tentativi"
        ) from last_err

    def _do_reset(self):
        # la vera e propria logica di reset, che può essere richiamata più volte in caso di timeout
        self.session.hard_reset()
        # timeout più corto del watchdog: se i menu si incastrano usciamo con
        # TimeoutError (recuperabile) invece di aspettare il SIGKILL del watchdog
        self.session.advance_to_in_game(timeout=self.advance_timeout_s)

        # input neutro mentre si aspetta il primo frame valido: con blocking_input
        # Dolphin si ferma ad aspettare l'input a ogni frame in game
        gs = None
        while gs is None:
            self.session.apply_input(player_idx=self.agent_idx)
            gs = self.session.step()

        self._prev_gs = gs
        self._steps = 0
        self.session.old_stocks = list(self.session.stocks)
        obs = self._build(gs, self.ctx)
        self._last_obs = obs

        assert obs.shape == self.observation_space.shape, (obs.shape, self.observation_space.shape)

        return obs, {}

    def step(self, action):
        try:
            return run_with_watchdog(
                lambda: self._do_step(action),
                timeout_s=self.step_timeout_s,          # 10-15s: enorme per uno step reale (ms), scatta solo se qualcosa è andato storto
                on_timeout=self.session.force_kill_dolphin,
            )
        
        except RECOVERABLE_ERRORS as e:                 # timeout, dolphin morto, ecc.
            self._log(f"step: {type(e).__name__}, chiudo la sessione e tronco l'episodio")
            self.session.close()
            # truncated=True fa scattare l'auto-reset del VecEnv, che rifà hard_reset (reboot pulito)
            return self._last_obs, 0.0, False, True, {"timeout": True, "P1_stocks": 0, "P2_stocks": 0, "P1_percent": 0, "P2_percent": 0}

    def _do_step(self, action):
        reward = 0.0
        terminated = False
        gs = None  # ultimo gamestate valido visto nello skip

        (x, y), button = self._decode(action)

        for k in range(self.frame_skip):
            self.session.apply_input(player_idx=self.agent_idx, button=button,
                                     stick_x=x, stick_y=y, press=(k == 0))

            g = self.session.step()

            if g is None or g.frame < 0:
                continue

            if self.debug:
                print(f"[step] inst={self.session.config.instance_id} frame={g.frame}", flush=True)

            gs = g

            if self.session.match_over:
                won = gs.players[self.ctx.opp_port].stock == 0
                reward += self.win_bonus if won else -self.win_bonus
                terminated = True
                break

            reward += self._reward(self._prev_gs, gs, self.ctx)
            self._prev_gs = gs

        self._steps += 1  # conta DECISIONI, non frame di gioco (coerente col //frame_skip in max_steps)

        stocks = self.session.stocks
        percents = self.session.percents
        if len(stocks) < 2 or len(percents) < 2:   # gamestate assente/crashato: 0 per entrambi
            stocks, percents = [0, 0], [0, 0]

        info = {
            "P1_stocks": stocks[0], "P2_stocks": stocks[1],
            "P1_percent": percents[0], "P2_percent": percents[1],
        }

        if terminated:
            self._log(f"episodio terminato @ frame {gs.frame} dopo {self._steps} decisioni")

        if gs is None:
            # nessun frame valido in tutto lo skip (raro): stato invariato
            return self._last_obs, reward, terminated, False, info

        obs = self._build(gs, self.ctx)
        self._last_obs = obs

        if terminated:   # lascia finalizzare il .slp (GAME_END + metadata) prima che il reset uccida Dolphin
            self.session.drain_to_game_end(
                apply_input_fn=lambda: self.session.apply_input(player_idx=self.agent_idx))

        truncated = (not terminated) and (self._steps >= self.max_steps)
        return obs, reward, terminated, truncated, info
    
    def close(self):
        if hasattr(self, "session"):
            self.session.close()
