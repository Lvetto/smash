from __future__ import annotations
import os
import subprocess
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


def kill_dolphin() -> None:
    """
    Uccide eventuali processi Dolphin appesi, che altrimenti tengono occupata la porta dello spectator server (51441).
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
    il display virtuale.
    """

    dolphin_exe: Path           # path diretto a .../usr/bin/dolphin-emu. NON AppImage o directory di installazione.
    iso: Path                   # immagine del gioco .iso (NON .rvz o .ciso)
    replay_dir: Path            # directory dove salvare i replay
    display_size: tuple = (640, 480)
    use_exi_inputs: bool = True # True per addestramento. EXI supporta aumento della velocità
    enable_ffw: bool = True     # True per addestramento. FFWD abilita aumento della velocità
    headless: bool = True       # True = crea un Xvfb; False = usa il DISPLAY esistente

    # -- identità dell'istanza (per il multi-istanza) --
    slippi_port: int = 31241    # porta dello spectator server; unica per istanza
    instance_id: int = 0        # solo per logging/debug
    use_instant_restart: bool = True  # prova il restart veloce; fallback = navigazione menu

    infinite_time: bool = False  # True = tempo infinito. False = tempo normale di 8 minuti
    instant_match_restart: bool = False  # True = restart istantaneo. False = restart lento via menu. Messo a False perché è implementato male e crea problemi con la selezione di personaggi e scenari

    @classmethod
    def from_env(cls, dotenv_path: Optional[str] = None, **overrides) -> "MeleeConfig":
        """Costruisce la config leggendo le variabili dal .env.

        Variabili attese:
          DOLPHIN_EXI_DIR  -> path diretto a dolphin/usr/bin/dolphin-emu
          SMBM_ISO_PATH    -> path (relativo alla cwd) dell'immagine .iso
          REPLAY_DIR       -> opzionale; default /home/luca/melee/replays
        """

        load_dotenv(dotenv_path)

        dolphin_exe = Path(os.environ["DOLPHIN_EXI_DIR"])
        iso = Path(os.environ["SMBM_ISO_PATH"])
        # get() ritorna None se non settata: Path(None) esplode -> forniamo un default.
        replay_dir = Path(os.environ.get("REPLAY_DIR", "/home/luca/melee/replays"))

        cfg = cls(dolphin_exe=dolphin_exe, iso=iso, replay_dir=replay_dir, **overrides)
        cfg.validate()

        return cfg

    '''@classmethod
    def for_instance(self, i: int) -> "MeleeConfig":
        """
        Ritorna una COPIA della config specializzata per l'istanza i:
        porta = 51441 + i, instance_id = i. La replay_dir resta condivisa
        (i replay hanno nomi univoci), la home di Dolphin è isolata a runtime
        via tmp_home_directory=True nel Console.
        """
        return replace(self, slippi_port=51441 + i, instance_id=i)'''
    
    @classmethod
    def for_instance(cls, i):
        base = cls.from_env()          # o cls() se ha default sensati
        return replace(base, slippi_port=51441 + i, instance_id=i, replay_dir=base.replay_dir / f"instance_{i}")

    def validate(self) -> None:
        """Verifica che i path esistano. Lancia FileNotFoundError se qualcosa non va."""
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
    """

    character: melee.Character = melee.Character.FOX
    cpu_level: int = 9

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
    """

    def __init__(self,
        config: Optional[MeleeConfig] = None,
        players: Optional[Sequence[PlayerSpec]] = None,
        console: Optional[melee.Console] = None):

        self.config = config or MeleeConfig.from_env()
        self.players = players or test_player_specs()

        self._external_console = console    # se esiste, non creiamo una nuova console ma usiamo quello esterno. Utile per il multi-istanza.
        self.console: Optional[melee.Console] = console
        self.controllers: list = []

        self.display: Optional[Display] = None

        self._gamestate = None
        self._in_game_frames = 0

        self.old_stocks = [4 for _ in self.players]  # per rilevare il match_over
    

    # -- costruzione risorse --

    def _ensure_display(self) -> None:
        """Crea il display virtuale una sola volta; sopravvive agli hard reset."""

        if not self.config.headless:
            self.display = None
            return
        
        if self.display is None:
            self.display = Display(visible=0, size=self.config.display_size)
            self.display.start()

    def _build_console(self) -> None:
        """(Ri)crea l'oggetto Console e i controller. Non fa boot: quello è run()."""

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
                disable_audio=True,
                infinite_time=self.config.infinite_time,
                instant_match_restart=self.config.instant_match_restart,
            )

        self.controllers = [
            melee.Controller(self.console, port + 1) for port in range(len(self.players))
        ]

    # -- gestione della sessione --

    def hard_reset(self):
        """
        Riavvia la sessione di gioco, uccidendo eventuali processi Dolphin appesi e ricreando il display virtuale e la console.
        """

        if self.console is not None and self._external_console is None:
            try:
                self.console.stop()
            except Exception as e:
                print(f"[melee_env] errore stop console pre-hard_reset: {e}")
            self.console = None

        self._ensure_display()
        self._build_console()

        self.console.run(iso_path=str(self.config.iso))
        assert self.console.connect(), "connessione a Dolphin fallita"

        for controller in self.controllers:
            #print(f"[melee_env] Collegamento controller porta {controller.port}...")
            controller.connect()

        self._gamestate = None
        self._in_game_frames = 0
        return self

    def soft_reset(self, timeout=None):
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
        timeout: Optional[float] = None,
        dbg: bool = False):
        """
        Avanza la partita fino a entrare in game, navigando i menu per selezionare i giocatori e lo stage.
        Ritorna il gamestate finale (in game). Se timeout è specificato, lancia TimeoutError se non si entra in game entro il tempo specificato.
        """

        start = time.time()
        menu_helper = melee.MenuHelper()
        specs = list(player_specs or self.players)
        last_idx = len(specs) - 1

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

            for i, spec in enumerate(specs):
                menu_helper.menu_helper_simple(
                    gamestate=gs,
                    controller=self.controllers[i],
                    character_selected=spec.character,
                    stage_selected=stage or melee.Stage.FINAL_DESTINATION,
                    cpu_level=spec.cpu_level,
                    autostart=(i == last_idx),  # l'ultimo giocatore avvia
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
        return (
            self._gamestate is not None
            and self._gamestate.menu_state == melee.Menu.IN_GAME
            and not self.match_over
        )

    @property
    def positions(self) -> list:
        if not self.is_in_match:
            return []
        gs = self._gamestate
        return [(gs.players[i + 1].position.x, gs.players[i + 1].position.y)
                for i in range(len(self.players))]

    @property
    def stocks(self) -> list:
        gs = self._gamestate
        if gs.menu_state != melee.Menu.IN_GAME:
            return []
        return [gs.players[i + 1].stock for i in range(len(self.players))]

    @property
    def percents(self) -> list:
        if not self.is_in_match:
            return []
        gs = self._gamestate
        return [gs.players[i + 1].percent for i in range(len(self.players))]

    # -- input controller --

    @property
    def controller_ports(self) -> list:
        return [controller.port for controller in self.controllers]

    def press_button(self, player_idx: int, button: melee.Button):
        if player_idx < 0 or player_idx >= len(self.controllers):
            raise ValueError(
                f"player_idx {player_idx} fuori range (0..{len(self.controllers) - 1})"
            )
        self.controllers[player_idx].press_button(button)

    def apply_input(self, player_idx, button=None, stick_x=0.5, stick_y=0.5, press=True):
        # dovrebbe andare bene per tutti i comandi che durano un singolo frame, a patto di avere un frame_skip >= 2
        # NON va bene per L e R (scudi, dash, grab) perché devono essere tenuti per più frame. Per ora non ho voglia di implementarli
        c = self.controllers[player_idx]
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
            try:
                self.console.stop()
            except Exception as e:
                print(f"[melee_env] errore durante lo stop della console: {e}")
        self.console = None

        if self.display is not None:
            try:
                self.display.stop()
            except Exception as e:
                print(f"[melee_env] errore durante lo stop del display: {e}")
        self.display = None

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


# -- multithreading ---

class Timeout(Exception):
    pass

# funzione per evitare che il training rimanga appeso ad aspettare processi bloccati. Lancia Timeout se la funzione fn() non ritorna entro timeout_s secondi.
def run_with_watchdog(fn, timeout_s, on_timeout):
    """
    Lancia la funzione fn() in un thread separato e attende al massimo timeout_s secondi.
    Se fn() non ritorna entro il timeout, chiama on_timeout() e lancia Timeout. Se fn() ritorna dopo il timeout, lancia comunque Timeout.
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
            raise Timeout() from e
        raise                              # eccezione vera, non nostra
    finally:
        finished.set()
        w.join(timeout=1.0)

    if timed_out["v"]:                     # fn è tornata dopo il kill
        raise Timeout()
    return result


if __name__ == "__main__":
    cfg = MeleeConfig.from_env()
    print(f"[melee_env] Config caricata: {cfg}")

    with MeleeSession(config=cfg) as session:

        print("[melee_env] Sessione avviata")

        session.advance_to_in_game(dbg=True)
        print("\n[melee_env] Partita avviata")

        print(f"[melee_env] is_in_match: {session.is_in_match}")
        print(f"[melee_env] positions: {session.positions}")
        print(f"[melee_env] stocks: {session.stocks}")
        print(f"[melee_env] percents: {session.percents}")
        print(f"[melee_env] controller_ports: {session.controller_ports}")

        session.apply_input(player_idx=0, button=melee.Button.BUTTON_A, stick_x=0.0, stick_y=1.0)
        print("[melee_env] Input applicato: player 0, A + stick up")

        print(f"[melee_env] is_in_match: {session.is_in_match}")

        prev_stocks = None
        while True:
            gs = session.step()
            if gs is None:
                continue
            stocks = session.stocks
            percents = session.percents
            print(f"[melee_env] stocks: {stocks}, percents: {percents}\t\t\t", end="\r")
            if session.match_over:
                print(f"\n[melee_env] RESTART rilevato @ frame {gs.frame}: {prev_stocks} -> {stocks}")
                break
            prev_stocks = stocks
        
    print("[melee_env] close() eseguito dal context manager, sessione terminata.")
