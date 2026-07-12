from __future__ import annotations
import configparser
import errno
import fcntl
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence
from dotenv import load_dotenv
from pyvirtualdisplay import Display
import melee
from random import Random
import psutil
import threading
import numpy as np

# per hard reset all'inizio del training. I processi dolphin hanno la tendenza a rimanere appesi
def kill_dolphin() -> None:
    """
    Uccide eventuali processi Dolphin appesi, che altrimenti tengono occupata la porta dello spectator server (51441).
    ATTENZIONE: uccide TUTTI i dolphin-emu della macchina, anche quelli di altri run in corso.
    """
    subprocess.run(
        ["pkill", "-f", "dolphin-emu"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# -- configurazione per la sessione di gioco --

@dataclass
class MeleeConfig:
    """
    Configurazione di base per la sessione Melee.

    Le tre risorse che DEVONO essere uniche per istanza (per la parallelizzazione)
    sono: la porta Slippi, la home di Dolphin (isolata via tmp_home_directory) e
    il display virtuale (pyvirtualdisplay ne assegna uno libero per processo).
    """

    dolphin_exe: Path           # path diretto a .../usr/bin/dolphin-emu. NON AppImage o directory di installazione.
    iso: Path                   # immagine del gioco .iso (NON .rvz o .ciso)
    replay_dir: Path            # directory dove salvare i replay
    display_size: tuple = (640, 480)    # dimensioni del display virtuale (Xvfb). Più grande = più lento. Più piccolo = più probabile che Dolphin crashi.
    use_exi_inputs: bool = True # True per addestramento. EXI supporta aumento della velocità
    enable_ffw: bool = True     # True per addestramento. FFWD abilita aumento della velocità
    blocking_input: bool = True # True = Dolphin si ferma ad aspettare l'input del bot a ogni frame.
                                # INDISPENSABILE col multi-istanza: senza (default libmelee = False), con FFW
                                # Dolphin corre libero mentre il worker aspetta il lockstep del VecEnv e
                                # console.step() legge frame sempre più vecchi (obs stantie, action in ritardo).
    headless: bool = True       # True = crea un Xvfb; False = usa il DISPLAY esistente
    disable_audio: bool = True  # False = audio attivo (per giocare dal vivo contro l'agente)
    human_pad_config: Optional[Path] = None  # template ini col mapping del gamepad umano
                                             # (generato con `python play.py --configure-pad`)

    # -- identità dell'istanza (per il multi-istanza) --
    slippi_port: int = 31241    # porta dello spectator server; unica per istanza
    instance_id: int = 0        # solo per logging/debug
    use_instant_restart: bool = True  # prova il restart veloce; fallback = navigazione menu

    # -- opzioni di gioco (meglio non toccare, sono così perché funzionano) --
    infinite_time: bool = False  # True = tempo infinito. False = tempo normale di 8 minuti
    instant_match_restart: bool = False  # True = restart istantaneo. False = restart lento via menu. Messo a False perché è implementato male e crea problemi con la selezione di personaggi e scenari

    @staticmethod
    def _resolve_dolphin_exe(path: Path) -> Path:
        """Accetta sia il path diretto all'eseguibile sia la radice di un AppImage estratto."""
        if path.is_dir():
            candidate = path / "usr" / "bin" / "dolphin-emu"
            if candidate.is_file():
                return candidate
        return path

    @classmethod
    def from_env(cls, dotenv_path: Optional[str] = None, save_name: str = "default",
                 dolphin_env_var: str = "DOLPHIN_EXI_DIR", **overrides) -> "MeleeConfig":
        """Costruisce la config leggendo le variabili dal .env.

        Variabili attese:
          DOLPHIN_EXI_DIR  -> path diretto a dolphin/usr/bin/dolphin-emu
                              (o un'altra build via dolphin_env_var, es. DOLPHIN_NETPLAY_DIR)
          SMBM_ISO_PATH    -> path (relativo alla cwd) dell'immagine .iso
          REPLAY_DIR       -> opzionale; default /home/luca/melee/replays
        """

        load_dotenv(dotenv_path)

        dolphin_exe = cls._resolve_dolphin_exe(Path(os.environ[dolphin_env_var]))
        iso = Path(os.environ["SMBM_ISO_PATH"])
        replay_dir = Path(os.environ["REPLAY_DIR"]) / save_name


        cfg = cls(dolphin_exe=dolphin_exe, iso=iso, replay_dir=replay_dir, **overrides)
        cfg.validate()

        return cfg

    @classmethod
    def for_instance(cls, i: int, save_name: str = "default") -> "MeleeConfig":
        """
        Ritorna una config per l'istanza i: porta Slippi unica (51441 + i) e replay_dir dedicata
        """
        base = cls.from_env(save_name=save_name)   # per poter usare le variabili d'ambiente e i path relativi partiamo da una config base. Poi la modifichiamo per l'istanza i.
        cfg = replace(base, slippi_port=51441 + i, instance_id=i,
                      replay_dir=base.replay_dir / f"instance_{i}")
        cfg.replay_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    @classmethod
    def for_play(cls, save_name: str = "play", **overrides) -> "MeleeConfig":
        """
        Config per giocare in tempo reale contro l'agente. Usa la build netplay
        (DOLPHIN_NETPLAY_DIR nel .env): quella EXI del training forza il backend
        video Null e non mostra nulla. Velocità normale, video e audio attivi,
        pipe non bloccanti (il ritmo lo dà Dolphin a 60fps, non il bot).
        """
        defaults = dict(headless=False, enable_ffw=False, use_exi_inputs=False,
                        blocking_input=False, disable_audio=False)
        defaults.update(overrides)
        return cls.from_env(save_name=save_name, dolphin_env_var="DOLPHIN_NETPLAY_DIR",
                            **defaults)

    def validate(self) -> None:
        """Verifica che la config abbia senso e che le risorse esistano. Lancia FileNotFoundError se qualcosa non va."""

        if not self.dolphin_exe.is_file():
            raise FileNotFoundError(f"eseguibile Dolphin non trovato: {self.dolphin_exe}")
        
        if not self.iso.is_file():
            raise FileNotFoundError(f"ISO non trovata: {self.iso}")
        
        if self.iso.suffix.lower() == ".rvz":
            print(
                "[melee_env] ATTENZIONE: stai usando un .rvz. Se il boot fallisce, "
                "converti in .iso con dolphin-tool convert -f iso."
            )

        self.replay_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PlayerSpec:
    """
    Configurazione di un giocatore (porta) per la partita. Contiene il personaggio e il livello CPU.
    Livello CPU 0 = player, 1-9 = CPU, 10 = random.
    human=True = porta riservata a un giocatore umano: niente pipe libmelee, l'input arriva
    dal gamepad configurato in Dolphin (config.human_pad_config) e nei menu sceglie da sé.
    """

    character: melee.Character = melee.Character.FOX
    cpu_level: int = 9
    human: bool = False


def _is_human(spec) -> bool:
    """True se lo spec (eventualmente None nei test) è una porta umana."""
    return spec is not None and getattr(spec, "human", False)

# usato per testare la sessione di gioco. Inizializza due cpu, quindi una partita autonoma che non richiede input
def test_player_specs() -> list:
    """FOX vs MARTH, entrambe CPU livello 9"""
    return [
        PlayerSpec(melee.Character.FOX, 9),
        PlayerSpec(melee.Character.MARTH, 9),
    ]

# -- gestione di una sessione di gioco

class MeleeSession:
    """
    Gestisce una sessione di gioco Melee con Dolphin/Slippi, con due controller (porte) e un display virtuale opzionale.

    Supporta multi-processing: ogni istanza ha una config unica che dovrebbe evitare conflitti. Un watchdog verificia l'istanza sia ancora viva, e la ricrea in caso di crash.
    Se il watchdog rileva un timeout, uccide Dolphin e lancia Timeout. Se qualcosa va storto, lancia un'eccezione invece di restare appeso.
    """

    # timeout di default per le fasi di boot
    connect_timeout_s = 30.0        # attesa dello spectator server di Dolphin
    controller_timeout_s = 30.0     # attesa dell'apertura delle pipe da parte di Dolphin

    def __init__(self,
        config: Optional[MeleeConfig] = None,
        players: Optional[Sequence[PlayerSpec]] = None,
        console: Optional[melee.Console] = None):

        self.config = config or MeleeConfig.from_env()  # config passata o defalt dal .env
        self.players = players or test_player_specs()   # default = due CPU livello 9 (partita autonoma)

        self._external_console = console    # se esiste, non creiamo una nuova console ma usiamo quello esterno. Utile per il multi-istanza.
        self.console: Optional[melee.Console] = console
        self.controllers: list = []

        self.display: Optional[Display] = None

        self._gamestate = None
        self._in_game_frames = 0

        self.old_stocks = [4 for _ in self.players]  # per rilevare il match_over

    def _log(self, msg: str) -> None:       # logga lo stato dell'istanza. Utile con più processi perché i log sono scritti su file, separati per istanza.
        print(f"[melee_env inst={self.config.instance_id}] {msg}", flush=True)

    # -- costruzione risorse --

    def _ensure_display(self) -> None:      # basta che il display virtuale esista. Può essere creato una singola volta e non dovrebbe essere interessato ai crash del resto del codice
        """Crea il display virtuale una sola volta; sopravvive agli hard reset."""

        if not self.config.headless:
            self.display = None
            return

        if self.display is None:
            self.display = Display(visible=0, size=self.config.display_size)
            self.display.start()

    def _build_console(self) -> None:
        """(Ri)crea l'oggetto Console e i controller"""

        if self._external_console is not None:  # se ci hanno passato un console esterno, lo usiamo senza crearne uno nuovo
            self.console = self._external_console

        else:
            self.console = melee.Console(
                path=str(self.config.dolphin_exe),
                replay_dir=str(self.config.replay_dir),
                slippi_port=self.config.slippi_port,
                tmp_home_directory=True,
                use_exi_inputs=self.config.use_exi_inputs,
                enable_ffw=self.config.enable_ffw,
                blocking_input=self.config.blocking_input,
                disable_audio=self.config.disable_audio,
                infinite_time=self.config.infinite_time,
                instant_match_restart=self.config.instant_match_restart,
            )

        # la lista resta allineata a self.players: None per le porte umane (niente pipe)
        self.controllers = [
            None if _is_human(spec) else melee.Controller(self.console, port + 1)
            for port, spec in enumerate(self.players)
        ]

    def _dolphin_alive(self) -> bool:       # verifica il processo sia ancora attivo. Se il processo è morto, la pipe del controller non si sblocca e il worker resta appeso per sempre.
        proc = self.console._process if self.console is not None else None
        return proc is not None and proc.poll() is None

    def _kill_stale_dolphins(self) -> None:
        """
        Si assicura non rimangano in vita processi Dolphin dopo un crash/reset.
        """
        tmp_root = tempfile.gettempdir()
        port_marker = f"spectatorlocalport = {self.config.slippi_port}"
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                if "dolphin-emu" not in (p.info["name"] or ""):
                    continue
                cmd = p.info["cmdline"] or []
                if "-u" not in cmd:
                    continue
                home = cmd[cmd.index("-u") + 1]
                if not home.startswith(os.path.join(tmp_root, "libmelee_")):
                    continue  # non è un dolphin lanciato da libmelee: non lo tocchiamo
                ini = Path(home) / "Config" / "Dolphin.ini"
                if not ini.is_file():
                    continue
                # il nome della chiave varia tra build (SpectatorLocalPort /
                # SlippiSpectatorLocalPort): il match sul suffisso copre entrambe
                if port_marker in ini.read_text(errors="ignore").lower():
                    self._log(f"ucciso dolphin stantio pid={p.pid} sulla porta {self.config.slippi_port}")
                    p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, OSError):
                continue

    def _connect_controller_safe(self, controller, timeout: Optional[float] = None) -> None:
        """
        Connessione di un controller, non bloccante con open().
        """
        timeout = timeout or self.controller_timeout_s
        deadline = time.time() + timeout
        while True:
            try:
                fd = os.open(controller.pipe_path, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError as e:
                if e.errno != errno.ENXIO:  # ENXIO = nessun lettore sulla FIFO: Dolphin non è ancora pronto
                    raise
            if not self._dolphin_alive():
                raise RuntimeError(
                    f"Dolphin (porta {self.config.slippi_port}) è morto prima di aprire "
                    f"la pipe del controller {controller.port}"
                )
            if time.time() > deadline:
                raise TimeoutError(
                    f"Dolphin non ha aperto la pipe del controller {controller.port} "
                    f"entro {timeout}s"
                )
            time.sleep(0.1)

        # ripristina la modalità bloccante
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)

        # replica ciò che fa Controller.connect() su Linux
        controller.pipe = os.fdopen(fd, "w")
        self.console.controllers.append(controller)

    def _install_human_pad_config(self) -> None:
        """
        Installa il mapping del gamepad umano nella home temporanea di Dolphin, per le
        porte con PlayerSpec.human=True: la sezione [GCPad<porta>] in GCPadNew.ini e
        SIDevice<porta-1> = 6 (standard controller) in Dolphin.ini. Va chiamata tra la
        creazione della console (che scrive le config) e console.run() (che le legge).
        """
        human_ports = [i + 1 for i, spec in enumerate(self.players) if _is_human(spec)]
        if not human_ports:
            return

        template_path = getattr(self.config, "human_pad_config", None)
        if template_path is None or not Path(template_path).is_file():
            raise FileNotFoundError(
                "manca il mapping del gamepad umano: generalo con "
                "`python play.py --configure-pad` e passalo in MeleeConfig.human_pad_config"
            )

        template = configparser.ConfigParser()
        template.optionxform = str   # i nomi delle chiavi di Dolphin vanno preservati
        template.read(template_path)
        if not template.sections():
            raise ValueError(f"nessuna sezione di mapping in {template_path}")
        mapping = template[template.sections()[0]]   # unica sezione attesa, il nome è irrilevante

        config_dir = Path(self.console._get_dolphin_config_path())

        pads = configparser.ConfigParser()
        pads.optionxform = str
        pads.read(config_dir / "GCPadNew.ini")
        core = configparser.ConfigParser()
        core.optionxform = str
        core.read(config_dir / "Dolphin.ini")

        for port in human_ports:
            section = f"GCPad{port}"
            if pads.has_section(section):
                pads.remove_section(section)
            pads.add_section(section)
            for key, value in mapping.items():
                pads.set(section, key, value)
            core.set("Core", f"SIDevice{port - 1}", melee.enums.ControllerType.STANDARD.value)

        with open(config_dir / "GCPadNew.ini", "w") as f:
            pads.write(f)
        with open(config_dir / "Dolphin.ini", "w") as f:
            core.write(f)
        self._log(f"mapping del pad umano installato (porte {human_ports})")

    def _safe_stop_console(self) -> None:
        """
        Ferma la console in maniera non bloccante, evitando close() che può restare appeso se Dolphin è morto. Chiude la pipe del controller e uccide Dolphin.
        """
        if self.console is None:
            return

        self.force_kill_dolphin()

        slippstream = self.console._slippstream if self.console is not None else None
        buf = slippstream._buffer if slippstream is not None else None
        if buf is not None:
            deadline = time.time() + 10.0
            try:
                while time.time() < deadline and buf.poll(0.2):
                    buf.recv_bytes()
            except (EOFError, OSError):
                pass  # pipe chiusa dal figlio: va benissimo

        try:
            self.console.stop()
        except Exception as e:
            self._log(f"errore durante lo stop della console: {e}")
        self.console = None

    # -- gestione della sessione --

    def hard_reset(self):
        """
        Riavvia la sessione di gioco, uccidendo eventuali processi Dolphin appesi e ricreando il display virtuale e la console.
        Ogni fase è fail-fast: in caso di problemi lancia un'eccezione invece di bloccarsi.
        """

        if self.console is not None and self._external_console is None:
            self._safe_stop_console()

        if self._external_console is None:
            self._kill_stale_dolphins()   # la porta slippi deve essere libera prima del boot

        self._ensure_display()
        self._build_console()
        self._install_human_pad_config()   # prima del run: Dolphin legge la config al boot

        self.console.run(iso_path=str(self.config.iso))
        self._log(f"dolphin avviato (porta {self.config.slippi_port})")

        # console.connect() prova per 10s e ritorna False in caso di fallimento:
        if not self.console.connect():
            alive = "vivo" if self._dolphin_alive() else "MORTO"
            raise RuntimeError(
                f"connessione a Dolphin fallita (porta {self.config.slippi_port}, processo {alive})"
            )

        for controller in self.controllers:
            if controller is not None:   # le porte umane non hanno pipe
                self._connect_controller_safe(controller)
        self._log("console e controller connessi")

        self._gamestate = None
        self._in_game_frames = 0
        return self

    def soft_reset(self, timeout: Optional[float] = 120.0):
        """Aspetta che la partita venga resettata (stock a 4) e ritorna il gamestate in game. Timeout di default = 120s."""
        start = time.time()
        while True:

            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError("timeout durante il restart")
            gs = self.console.step()

            if gs is None:
                continue

            self._gamestate = gs
            if gs.menu_state == melee.Menu.IN_GAME:
                break

        self._in_game_frames = 0
        return gs

    def advance_to_in_game(self,
        player_specs: Optional[Sequence[PlayerSpec]] = None,
        stage: Optional[melee.Stage] = None,
        timeout: Optional[float] = 120.0,
        dbg: bool = False):
        """
        Avanza la partita fino a entrare in game, navigando i menu per selezionare i giocatori e lo stage.
        Ritorna il gamestate finale (in game). Il timeout di default è 120s: la navigazione
        dei menu ogni tanto si incastra e senza timeout un worker parallelo resterebbe appeso per sempre.
        """

        start = time.time()
        menu_helper = melee.MenuHelper()
        specs = list(player_specs or self.players)
        bot_idxs = [i for i, spec in enumerate(specs) if not _is_human(spec)]
        # con un umano in partita nessun bot avvia: è l'umano a premere start quando è pronto
        autostart_idx = bot_idxs[-1] if bot_idxs and len(bot_idxs) == len(specs) else None

        while True:
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError("Timeout raggiunto durante l'avvio della partita")

            gs = self.console.step()
            if gs is None:
                # partita non ancora pronta: buttiamo via i frame None
                if dbg:
                    print("[melee_env] Attesa avvio partita...", end="\r")
                continue

            self._gamestate = gs
            if gs.menu_state == melee.Menu.IN_GAME:
                break

            for i in bot_idxs:
                spec = specs[i]
                menu_helper.menu_helper_simple(
                    gamestate=gs,
                    controller=self.controllers[i],
                    character_selected=spec.character,
                    stage_selected=stage or melee.Stage.FINAL_DESTINATION,
                    cpu_level=spec.cpu_level,
                    autostart=(i == autostart_idx),
                )

            if dbg:
                print(
                    f"[melee_env] Avvio: menu_state={gs.menu_state} frame={gs.frame} "
                    f"t={time.time() - start:.2f}s",
                    end="\r",
                )

        return self._gamestate

    def step(self):
        """
        Avanza di un frame e ritorna il gamestate (o None). Mantiene il contatore
        in_game_frames aggiornato.
        """
        gs = self.console.step()
        if gs is not None:
            self._gamestate = gs
            if gs.menu_state == melee.Menu.IN_GAME:
                self._in_game_frames += 1
        return gs

    def drain_to_game_end(self, apply_input_fn=None, max_frames: int = 1200,
                          timeout_s: float = 8.0) -> None:
        """
        Dopo la fine di un match, avanza i frame (input neutro) finché la partita
        esce da IN_GAME. Senza questa attesa l'hard_reset ucciderebbe Dolphin
        prima che scriva l'evento GAME_END e il blocco metadata del replay,
        lasciando .slp non finalizzati (raw di lunghezza 0, niente metadata).
        max_frames/timeout_s limitano l'attesa: timeout_s sta sotto il watchdog
        di step (12s), altrimenti scatterebbe un SIGKILL che vanifica il drain.
        """
        start = time.time()
        for _ in range(max_frames):
            if time.time() - start > timeout_s:
                break
            if apply_input_fn is not None:
                apply_input_fn()   # con blocking_input Dolphin aspetta l'input a ogni frame
            gs = self.step()
            if gs is None:
                continue
            if gs.menu_state != melee.Menu.IN_GAME:
                break               # schermata post-partita: GAME_END già scritto, replay finalizzato

    def _wait_fresh_match(self, timeout: Optional[float] = None) -> bool:
        """
        Attende che gli stock vengano resettati a 4 (nuovo match). Ritorna True se il match è stato resettato, False se timeout.
        """
        start = time.time()
        while True:

            # se timeout è specificato, ritorna False se il tempo è scaduto
            if timeout is not None and (time.time() - start) > timeout:
                return False

            # scartiamo tutti i frame fino all'avvio della partita
            gs = self.console.step()
            if gs is None:  # non dovrebbe succedere, ma se il gamestate è None, scartiamo il frame e riproviamo
                continue

            self._gamestate = gs
            if gs.menu_state != melee.Menu.IN_GAME:     # non dovrebbe succedere di vedere altre schermate, ma assicuriamoci di essere ancora in game
                continue

            stocks = [gs.players[i + 1].stock for i in range(len(self.players))]
            if all(stock == 4 for stock in stocks):     # match resettato, ritorniamo True
                return True

    # -- proprietà per lo stato della partita --

    def _player_values(self, getter) -> np.ndarray:
        """Applica getter a ogni giocatore dell'ultimo gamestate; array vuoto se non in game."""
        gs = self._gamestate
        if gs is None or gs.menu_state != melee.Menu.IN_GAME:
            return np.array([])
        return np.array([getter(gs.players[i + 1]) for i in range(len(self.players))])

    @property
    def match_over(self) -> bool:
        """
        Ritorna True se la partita è finita (un personaggio è morto).
        """
        if self._gamestate is None:
            return False

        a = any(stock <= 0 for stock in self.stocks)
        b = any(stock > old for stock, old in zip(self.stocks, self.old_stocks))
        self.old_stocks = self.stocks

        return a or b

    @property
    def is_in_match(self) -> bool:
        """ Ritorna True se siamo in partita (menu_state = IN_GAME e match non finito)"""
        return (
            self._gamestate is not None
            and self._gamestate.menu_state == melee.Menu.IN_GAME
            and not self.match_over
        )

    @property
    def positions(self) -> np.ndarray:
        """ Posizioni (x, y) dei giocatori, shape (n_players, 2). Array vuoto se non in game. """
        return self._player_values(lambda p: (p.position.x, p.position.y))

    @property
    def velocities(self) -> np.ndarray:
        """
        Velocità (vx_self, vy_self, vx_attack, vy_attack) dei giocatori, shape (n_players, 4).
        vx_self somma le componenti a terra e in aria (una delle due è sempre 0). Array vuoto se non in game.
        """
        return self._player_values(lambda p: (
            p.speed_ground_x_self + p.speed_air_x_self,
            p.speed_y_self,
            p.speed_x_attack,
            p.speed_y_attack,
        ))

    @property
    def stocks(self) -> np.ndarray:
        """ Stock dei giocatori, shape (n_players,). Array vuoto se non in game. """
        return self._player_values(lambda p: p.stock)

    @property
    def percents(self) -> np.ndarray:
        """ Percentuali di danno dei giocatori, shape (n_players,). Array vuoto se non in game. """
        return self._player_values(lambda p: p.percent)

    @property
    def distance(self) -> float:
        """ Distanza tra i due giocatori. 0.0 se non in game. """
        gs = self._gamestate
        if gs is None or gs.menu_state != melee.Menu.IN_GAME:
            return 0.0
        return gs.distance

    @property
    def facings(self):
        return self._player_values(lambda p: 1.0 if p.facing else -1.0)

    @property
    def hitstun_frames(self):
        return self._player_values(lambda p: p.hitstun_frames_left)
    
    @property
    def jumps_left(self):
        return self._player_values(lambda p: p.jumps_left)
    
    @property
    def on_ground(self):
        return self._player_values(lambda p: 1.0 if p.on_ground else 0.0)
    
    @property
    def off_stage(self):
        return self._player_values(lambda p: 1.0 if p.off_stage else 0.0)
    
    @property
    def invulnerable(self):
        return self._player_values(lambda p: 1.0 if p.invulnerable else 0.0)

    # -- input controller --

    @property
    def controller_ports(self) -> np.ndarray:
        return np.array([controller.port for controller in self.controllers
                         if controller is not None])

    def press_button(self, player_idx: int, button: melee.Button):
        if player_idx < 0 or player_idx >= len(self.controllers):
            raise ValueError(
                f"player_idx {player_idx} fuori range (0..{len(self.controllers) - 1})"
            )
        if self.controllers[player_idx] is None:
            raise ValueError(f"player_idx {player_idx} è una porta umana: nessun controller bot")
        self.controllers[player_idx].press_button(button)

    def apply_input(self, player_idx, button=None, stick_x=0.5, stick_y=0.5, press=True):
        # dovrebbe andare bene per tutti i comandi che durano un singolo frame, a patto di avere un frame_skip >= 2
        # NON va bene per L e R (scudi, dash, grab) perché devono essere tenuti per più frame. Per ora non ho voglia di implementarli
        c = self.controllers[player_idx]
        if c is None:
            raise ValueError(f"player_idx {player_idx} è una porta umana: nessun controller bot")
        c.tilt_analog(melee.Button.BUTTON_MAIN, stick_x, stick_y)  # stick: sempre tenuto
        if button is not None:
            if press:
                c.press_button(button)      # fronte di salita = attacco
            else:
                c.release_button(button)    # rilascio nei frame successivi
        c.flush()

    # -- gestione dell'istanza (chiusura, context manager) --

    def close(self) -> None:
        """
        Chiude la sessione di gioco, uccide Dolphin e chiude il display virtuale (se presente).
        """
        if self.console is not None and self._external_console is None:
            self._safe_stop_console()   # stesso anti-deadlock dell'hard_reset
        self.console = None

        if self.display is not None:
            try:
                self.display.stop()
            except Exception as e:
                self._log(f"errore durante lo stop del display: {e}")
        self.display = None

    # -- context manager --

    def __enter__(self) -> "MeleeSession":
        return self.hard_reset()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False  # non sopprime le eccezioni

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def force_kill_dolphin(self) -> None:
        console = self.console
        if console is None:
            return
        proc = getattr(console, "_process", None)
        if proc is None:
            return
        try:
            p = psutil.Process(proc.pid)
            for child in p.children(recursive=True):
                child.kill()
            p.kill()
        except psutil.NoSuchProcess:
            pass

class WatchdogTimeout(Exception):
    pass

# funzione per evitare che il training rimanga appeso ad aspettare processi bloccati. Lancia Timeout se la funzione fn() non ritorna entro timeout_s secondi.
def run_with_watchdog(fn, timeout_s, on_timeout):
    """
    Lancia la funzione fn() in un thread separato e attende al massimo timeout_s secondi.
    Se fn() non ritorna entro il timeout, chiama on_timeout() e lancia WatchdogTimeout. Se fn() ritorna dopo il timeout, lancia comunque WatchdogTimeout.
    """
    finished = threading.Event()
    timed_out = {"v": False}

    def _watch():
        if not finished.wait(timeout_s):   # scaduto
            print(f"[watchdog] TIMEOUT scattato dopo {timeout_s}s", flush=True)
            timed_out["v"] = True
            on_timeout()                   # SIGKILL sul Dolphin di questa sessione

    w = threading.Thread(target=_watch, daemon=True)
    w.start()

    try:
        result = fn()
    except Exception as e:
        if timed_out["v"]:                 # la recv è saltata perché abbiamo ucciso Dolphin
            raise WatchdogTimeout() from e
        raise                              # eccezione vera, non nostra
    finally:
        finished.set()
        w.join(timeout=1.0)

    if timed_out["v"]:                     # fn è tornata dopo il kill
        raise WatchdogTimeout()
    
    return result

