import gymnasium as gym
from gymnasium import spaces
import numpy as np
import melee
from lib import *


# definizione dell'ambiente per il training dell'agente

# -- tabelle di decodifica dell'azione --
STICK_MAP = {
    0: (0.5, 0.5),                      # neutro
    1: (0.5, 1.0), 2: (0.5, 0.0),       # su, giù
    3: (0.0, 0.5), 4: (1.0, 0.5),       # sx, dx
    5: (0.0, 1.0), 6: (1.0, 1.0),       # su-sx, su-dx
    7: (0.0, 0.0), 8: (1.0, 0.0),       # giù-sx, giù-dx
}
BUTTON_MAP = {
    0: None,
    1: melee.Button.BUTTON_A,   2: melee.Button.BUTTON_B,
    3: melee.Button.BUTTON_X,   4: melee.Button.BUTTON_Z,
    5: melee.Button.BUTTON_L,
}


STAGE_X_MAX = 85.0
STAGE_Y_MAX = 30.0

# -- definizione dell'ambiente Gymnasium per il training --
class MeleeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None,
                 agent_char=melee.Character.FOX,
                 opp_char=melee.Character.MARTH,
                 opp_level=2,
                 stage=melee.Stage.FINAL_DESTINATION):
        
        self.stage = stage
        self.agent_idx, self.opp_idx = 0, 1
        self.agent_port, self.opp_port = 1, 2   # le porte sono indicizzate a partire da 1 (porta 1 = controller reale, porta 2 = CPU)

        # porta 1 = agente (cpu_level=0 -> controller reale), porta 2 = CPU
        players = [PlayerSpec(agent_char, cpu_level=0),
                   PlayerSpec(opp_char, cpu_level=opp_level)]
        
        self.session = MeleeSession(config=config or MeleeConfig.from_env(), players=players)

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(24,), dtype=np.float32)
        #self.action_space = spaces.MultiDiscrete([9, 6])   # DQN non supporta spazi multi discreti. Appiattiamo quindi a un discreto singolo di 54 azioni (9 stick * 6 bottoni)
        self.action_space = spaces.Discrete(54)   # 9 stick * 6 bottoni

        self.gamestate = None
        self._prev = None
        self._steps = 0
        self._last_obs = np.zeros(24, dtype=np.float32)

        self._booted = False    # per distinguere il primo reset (boot) dai reset successivi (soft reset)

        # Parametri per il calcolo del reward
        #self.max_steps = 8 * 60 * 60 // 3   # abbastanza arbitrario, ma serve per evitare episodi troppo lunghi (8 minuti a 60 fps)
        #self.win_bonus = 100.0              # bonus di vittoria. Ancora da tarare, dovrebbe essere maggiore ma non troppo al reward totale per stock e danni in un episodio
        self.win_bonus = 2                # per DQN serve un reward comparabile agli altri.
        self.dmg_w     = 0.01               # bonus per danno. Per ogni stock vengono inflitti dell'ordine dei 100 danni, quindi peso di circa 0.01 lo rende comparabile al bonus per la rimozione di uno stock
        self.stock_w   = 1.0                # bonus per la rimozione di uno stock. Circa 100 volte quello per danno
    
        # Parametri per il frame skip e il numero massimo di passi
        self.frame_skip = 3
        self.max_steps = int(8 * 60 * 60 / self.frame_skip)  # 8 min * 60fps / skip = numero di decisioni

        # Per evitare blocchi in caso di processi appesi
        self.max_reset_attempts = 3
        self.reset_timeout_s = 90.0

        # per multiprocessing
        self._booted_once = False
        self.boot_stagger_s = 8.0   # secondi di sfasamento per istanza
        self.step_timeout_s = 12.0

    # -- API Gymnasium --

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if not self._booted_once:
            # istanza 0 → 0s, istanza 1 → 8s, ecc. Evita che due Dolphin
            # bootino nello stesso istante contendendosi CPU/X.
            time.sleep(self.session.config.instance_id * self.boot_stagger_s)
            self._booted_once = True

        last_err = None
        for attempt in range(self.max_reset_attempts):        # es. 3
            try:
                return run_with_watchdog(
                    self._do_reset,
                    timeout_s=self.reset_timeout_s,           # es. 90.0 (boot + menu con margine)
                    on_timeout=self.session.force_kill_dolphin,
                )
            except Timeout as e:
                last_err = e
                self.session.close()   # teardown pulito: console.stop() + display.stop(), azzera console→None
                time.sleep(2.0)        # margine per rilascio porta/risorse prima del re-boot
        raise RuntimeError(
            f"reset fallito dopo {self.max_reset_attempts} tentativi"
        ) from last_err
    
    def step(self, action):
        try:
            return run_with_watchdog(
                lambda: self._do_step(action),
                timeout_s=self.step_timeout_s,          # 10-15s: enorme per uno step reale (ms), scatta solo su hang
                on_timeout=self.session.force_kill_dolphin,
            )
        except Timeout:
            self.session.close()                        # console -> None, display fermato
            # truncated=True fa scattare l'auto-reset del VecEnv, che rifà hard_reset (reboot pulito)
            return self._last_obs, 0.0, False, True, {"timeout": True}

    def _do_step(self, action):
        stick_idx, button_idx = self._map_flat_action(action)
        x, y = STICK_MAP[stick_idx]
        button = BUTTON_MAP[button_idx]

        reward = 0.0
        terminated = False
        gs = None  # ultimo gamestate valido visto nello skip

        for k in range(self.frame_skip):
            # dovrebbe andare per ora, ma sarebbe da generalizzare un po' per supportare gli scudi, schivate e grab.
            # RICHIEDE frame_skip >= 2, altrimenti il fronte di discesa del bottone non viene catturato e l'input non viene registrato
            self.session.apply_input(
                player_idx=self.agent_idx, button=button,
                stick_x=x, stick_y=y, press=(k == 0),   # <-- edge solo sul primo frame
            )
            g = self.session.step()

            if g is None:
                continue  # frame di transizione: lo saltiamo senza contarlo

            print(f"[step] inst={self.session.config.instance_id} frame={g.frame}", flush=True)

            if g.frame < 0:   # countdown di inizio match: non c'è ancora uno stato valido
                continue  # frame di transizione: lo saltiamo senza contarlo

            gs = g

            if self.session.match_over:
                me  = gs.players[self.agent_port]
                opp = gs.players[self.opp_port]
                reward += self.win_bonus if opp.stock == 0 else -self.win_bonus
                terminated = True
                break  # match finito: interrompiamo lo skip

            # reward denso accumulato frame-per-frame; _snapshot aggiorna _prev
            # così il delta successivo resta frame-a-frame e la stock-loss guard vale ancora
            reward += self._compute_reward(gs)
            self._snapshot(gs)

        self._steps += 1  # conta DECISIONI, non frame di gioco (coerente col //frame_skip in max_steps)

        if gs is None:
            # nessun frame valido in tutto lo skip (raro): stato invariato
            return self._last_obs, reward, terminated, False, {}

        obs = self.build_observations(gs)
        self._last_obs = obs

        truncated = (not terminated) and (self._steps >= self.max_steps)
        return obs, reward, terminated, truncated, {}

    def close(self):
        if hasattr(self, "session"):
            self.session.close()

    # -- Utilità interne --

    def _map_flat_action(self, action):
        stick_idx = action // 6
        button_idx = action % 6
        return stick_idx, button_idx

    def build_observations(self, gs):
        p1, p2 = gs.players[self.agent_port], gs.players[self.opp_port]

        feats = []
        for p in (p1, p2):  # per entrambi i giocatori
            feats += [
                p.position.y / STAGE_Y_MAX,     # posizione normalizzata
                p.position.x / STAGE_X_MAX,     # posizione normalizzata 
                p.percent / 300.0,              # percentuale di danno normalizzata
                p.stock / 4.0,                  # numero di vite normalizzato
                1.0 if p.facing else -1.0,      # orientamento
                1.0 if p.on_ground else -1.0,   # stato a terra o in aria
            ]

            VEL_NORM = 5.0
            
            feats += [
                (p.speed_ground_x_self + p.speed_air_x_self) / VEL_NORM,
                p.speed_y_self / VEL_NORM,
                p.speed_x_attack / VEL_NORM,
                p.speed_y_attack / VEL_NORM,
            ]

        feats += [
            (p1.position.x - p2.position.x) / (2 * STAGE_X_MAX),    # distanza normalizzata tra i due giocatori
            (p1.position.y - p2.position.y) / (2 * STAGE_Y_MAX),    # distanza normalizzata tra i due giocatori
            gs.distance / 100.0,                                    # distanza normalizzata
            gs.frame / 28800.0,                                     # frame normalizzato
        ]

        obs = np.array(feats, dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)   # per stare nel box di osservazione
    
    def _compute_reward(self, gs):

        if gs.frame < 0:   # countdown di inizio match: non c'è ancora uno stato valido
            return 0.0

        if self._prev is None:  # solito check per evitare crash a metà addestramento
            return 0.0
        
        me,  opp  = gs.players[self.agent_port], gs.players[self.opp_port]
        d_opp_pct = int(opp.percent) - int(self._prev["opp_percent"])   # >0 = ho fatto danno (buono)
        d_my_pct  = int(me.percent)  - int(self._prev["my_percent"])    # >0 = ho subito danno (cattivo)
        d_opp_stk = int(self._prev["opp_stock"]) - int(opp.stock)       # >0 = ho tolto uno stock (buono)
        d_my_stk  = int(self._prev["my_stock"])  - int(me.stock)        # >0 = ho perso uno stock (cattivo)


        if d_opp_stk != 0: d_opp_pct = 0.0      # dopo la perdita di una vita, la percentuale di danno è resettata a 0
        if d_my_stk  != 0: d_my_pct  = 0.0      # dopo la perdita di una vita, la percentuale di danno è resettata a 0

        # check per evitare problemi
        if d_opp_stk < 0: d_opp_stk = 0; d_my_stk = 0   # stock aumentato = respawn/restart, non è una mia kill
        if d_my_stk  < 0: d_my_stk  = 0; d_opp_stk = 0   # stock aumentato = respawn/restart, non è una kill dell'avversario

        reward = self.dmg_w * (d_opp_pct - d_my_pct) + self.stock_w * (d_opp_stk - d_my_stk)

        return reward

    def _snapshot(self, gs):
        if gs.frame < 0:   # countdown di inizio match: non c'è ancora uno stato valido
            return None
        
        me = gs.players[self.agent_port]
        opp = gs.players[self.opp_port]
        self._prev = {                                              # stato precedente del match (per calcolare il reward)
            "my_percent": me.percent, "opp_percent": opp.percent,
            "my_stock": me.stock, "opp_stock": opp.stock,
        }

    def _do_reset(self):
        # la vera e propria logica di reset, che può essere richiamata più volte in caso di timeout
        self.session.hard_reset()
        self.session.advance_to_in_game()

        gs = self.session.step()
        while gs is None:
            gs = self.session.step()

        self._snapshot(gs)
        self._steps = 0
        self.session.old_stocks = list(self.session.stocks)
        obs = self.build_observations(gs)
        self._last_obs = obs
        return obs, {}

