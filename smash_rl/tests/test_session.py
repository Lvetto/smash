import fcntl
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import melee
import numpy as np
import psutil
import pytest

from smash_rl.session import (MeleeConfig, MeleeSession, PlayerSpec,
                              WatchdogTimeout, run_with_watchdog)
from smash_rl.tests.helpers import FakeSession, make_gs


def make_bare_session(**config_overrides):
    """MeleeSession senza Dolphin: config finta, nessuna console."""
    defaults = dict(instance_id=0, slippi_port=51441, headless=False,
                    iso=Path("/nonexistent/SSBM.iso"))
    return MeleeSession(config=SimpleNamespace(**{**defaults, **config_overrides}),
                        players=[None, None])


def make_console_stub(frame_seq):
    """Console finta il cui step() consuma la sequenza data."""
    seq = list(frame_seq)
    return SimpleNamespace(step=lambda: seq.pop(0) if seq else None,
                           controllers=[], _process=None)


def test_build_console_propagates_config(monkeypatch):
    # la console deve ricevere le opzioni dalla config, in particolare blocking_input=True:
    # il default di libmelee è False e con FFW farebbe correre Dolphin più veloce del worker
    # (obs stantie e action in ritardo quando il worker aspetta il lockstep del VecEnv)
    captured = {}

    class FakeConsole:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("smash_rl.session.melee.Console", FakeConsole)
    monkeypatch.setattr("smash_rl.session.melee.Controller",
                        lambda console, port: SimpleNamespace(port=port))

    cfg = MeleeConfig(dolphin_exe=Path("/nonexistent/dolphin-emu"),
                      iso=Path("/nonexistent/SSBM.iso"),
                      replay_dir=Path("/nonexistent/replays"),
                      slippi_port=51459)
    session = MeleeSession(config=cfg)
    session._build_console()

    assert captured["blocking_input"] is True
    assert captured["slippi_port"] == 51459
    assert captured["use_exi_inputs"] is cfg.use_exi_inputs
    assert captured["enable_ffw"] is cfg.enable_ffw
    assert len(session.controllers) == len(session.players)


@pytest.mark.dolphin
def test_melee_session():
    config = MeleeConfig.from_env()
    assert isinstance(config, MeleeConfig), "La configurazione dell'ambiente non è stata creata correttamente."
    
    with MeleeSession(config=config) as session:

        session.advance_to_in_game(dbg=True)

        assert session.is_in_match, "La partita non è iniziata correttamente."
        assert isinstance(session.positions, np.ndarray), "Le posizioni dei giocatori non sono disponibili."
        assert isinstance(session.stocks, np.ndarray), "Le vite dei giocatori non sono disponibili."
        assert isinstance(session.percents, np.ndarray), "I percentuali dei giocatori non sono disponibili."
        assert isinstance(session.controller_ports, np.ndarray), "Le porte dei controller non sono disponibili."

        session.hard_reset()
        session.advance_to_in_game(dbg=True)
        assert session.is_in_match, "La partita non è iniziata correttamente dopo il reset."


# -- MeleeConfig: variabili d'ambiente e risorse per istanza --

@pytest.fixture
def fake_dotenv(monkeypatch, tmp_path):
    exe = tmp_path / "dolphin-emu"
    exe.write_text("")
    iso = tmp_path / "SSBM.iso"
    iso.write_text("")
    monkeypatch.setenv("DOLPHIN_EXI_DIR", str(exe))
    monkeypatch.setenv("SMBM_ISO_PATH", str(iso))
    monkeypatch.setenv("REPLAY_DIR", str(tmp_path / "replays"))
    return tmp_path


def test_config_from_env(fake_dotenv):
    cfg = MeleeConfig.from_env(save_name="alpha")

    assert cfg.dolphin_exe == fake_dotenv / "dolphin-emu"
    assert cfg.iso == fake_dotenv / "SSBM.iso"
    assert cfg.replay_dir == fake_dotenv / "replays" / "alpha"
    assert cfg.replay_dir.is_dir()  # creata da validate()


def test_config_for_instance_unique_resources(fake_dotenv):
    # porta slippi e replay_dir DEVONO essere uniche per istanza: sono l'unica
    # cosa che separa i Dolphin paralleli
    cfg3 = MeleeConfig.for_instance(3, save_name="beta")
    cfg4 = MeleeConfig.for_instance(4, save_name="beta")

    assert cfg3.slippi_port == 51441 + 3
    assert cfg4.slippi_port == 51441 + 4
    assert cfg3.instance_id == 3
    assert cfg3.replay_dir == fake_dotenv / "replays" / "beta" / "instance_3"
    assert cfg3.replay_dir.is_dir()
    assert cfg3.replay_dir != cfg4.replay_dir


def test_config_validate_errors(fake_dotenv, tmp_path):
    with pytest.raises(FileNotFoundError, match="Dolphin"):
        MeleeConfig(dolphin_exe=tmp_path / "manca", iso=fake_dotenv / "SSBM.iso",
                    replay_dir=tmp_path / "r").validate()
    with pytest.raises(FileNotFoundError, match="ISO"):
        MeleeConfig(dolphin_exe=fake_dotenv / "dolphin-emu", iso=tmp_path / "manca.iso",
                    replay_dir=tmp_path / "r").validate()


# -- run_with_watchdog: l'unica difesa contro i worker appesi --

def test_watchdog_returns_result():
    called = []
    assert run_with_watchdog(lambda: 42, timeout_s=1.0, on_timeout=lambda: called.append(1)) == 42
    assert not called, "on_timeout non deve scattare se fn finisce in tempo"


def test_watchdog_timeout_kills_and_raises():
    killed = []
    with pytest.raises(WatchdogTimeout):
        run_with_watchdog(lambda: time.sleep(0.3), timeout_s=0.05,
                          on_timeout=lambda: killed.append(1))
    assert killed == [1]


def test_watchdog_wraps_exception_caused_by_kill():
    # il kill di Dolphin fa saltare la recv in corso: l'eccezione risultante
    # deve diventare WatchdogTimeout (recuperabile), non un errore fatale
    def fn():
        time.sleep(0.15)
        raise OSError("connessione saltata")

    with pytest.raises(WatchdogTimeout):
        run_with_watchdog(fn, timeout_s=0.05, on_timeout=lambda: None)


def test_watchdog_propagates_real_errors():
    def fn():
        raise ValueError("bug vero")

    killed = []
    with pytest.raises(ValueError):
        run_with_watchdog(fn, timeout_s=1.0, on_timeout=lambda: killed.append(1))
    assert not killed


# -- stato della partita --

def test_match_over_detects_zero_stocks_and_restart():
    session = FakeSession([])
    assert not session.match_over  # nessun gamestate

    session._gamestate = make_gs(p2_stock=0)
    assert session.match_over

    # stock che RISALE = restart della partita, va rilevato come fine episodio
    session._gamestate = make_gs(p1_stock=4, p2_stock=4)
    session.old_stocks = [3, 4]
    assert session.match_over


def test_is_in_match():
    session = FakeSession([])
    assert not session.is_in_match

    session._gamestate = make_gs()
    session.old_stocks = [4, 4]
    assert session.is_in_match

    menu = make_gs()
    menu.menu_state = melee.Menu.CHARACTER_SELECT
    session._gamestate = menu
    assert not session.is_in_match


def test_step_counts_only_in_game_frames():
    menu = make_gs()
    menu.menu_state = melee.Menu.CHARACTER_SELECT
    session = make_bare_session()
    session.console = make_console_stub([make_gs(frame=1), menu, None])

    assert session.step().frame == 1
    assert session._in_game_frames == 1
    assert session.step() is menu
    assert session._in_game_frames == 1, "i frame di menu non vanno contati"
    assert session.step() is None
    assert session._gamestate is menu, "un frame None non deve sporcare l'ultimo gamestate"


# -- navigazione menu e attese --

def test_advance_to_in_game_navigates_menus(monkeypatch):
    calls = []

    class FakeMenuHelper:
        def menu_helper_simple(self, **kwargs):
            calls.append((kwargs["controller"].port, kwargs["character_selected"],
                          kwargs["cpu_level"], kwargs["autostart"], kwargs["stage_selected"]))

    monkeypatch.setattr("smash_rl.session.melee.MenuHelper", FakeMenuHelper)

    menu = make_gs()
    menu.menu_state = melee.Menu.CHARACTER_SELECT
    ingame = make_gs()
    session = make_bare_session()
    session.players = [PlayerSpec(melee.Character.FOX, 0), PlayerSpec(melee.Character.MARTH, 9)]
    session.controllers = [SimpleNamespace(port=1), SimpleNamespace(port=2)]
    session.console = make_console_stub([None, menu, ingame])

    gs = session.advance_to_in_game(timeout=5.0)

    assert gs is ingame
    # per il frame di menu: un input per giocatore, autostart solo sull'ultimo,
    # cpu_level preso dallo spec del giocatore (0 = agente, 9 = CPU)
    assert calls == [
        (1, melee.Character.FOX, 0, False, melee.Stage.FINAL_DESTINATION),
        (2, melee.Character.MARTH, 9, True, melee.Stage.FINAL_DESTINATION),
    ]


def test_advance_to_in_game_times_out(monkeypatch):
    monkeypatch.setattr("smash_rl.session.melee.MenuHelper",
                        type("MH", (), {"menu_helper_simple": lambda self, **kw: None}))
    session = make_bare_session()
    session.players = [PlayerSpec(), PlayerSpec()]
    session.controllers = [SimpleNamespace(port=1), SimpleNamespace(port=2)]
    session.console = SimpleNamespace(step=lambda: None)  # mai in game

    with pytest.raises(TimeoutError):
        session.advance_to_in_game(timeout=0.05)


def test_soft_reset_waits_for_in_game():
    menu = make_gs()
    menu.menu_state = melee.Menu.POSTGAME_SCORES
    ingame = make_gs(frame=7)
    session = make_bare_session()
    session._in_game_frames = 123
    session.console = make_console_stub([None, menu, ingame])

    assert session.soft_reset(timeout=5.0) is ingame
    assert session._in_game_frames == 0

    session.console = SimpleNamespace(step=lambda: None)
    with pytest.raises(TimeoutError):
        session.soft_reset(timeout=0.05)


def test_wait_fresh_match():
    stale = make_gs(p1_stock=3)          # match precedente ancora in corso
    menu = make_gs()
    menu.menu_state = melee.Menu.POSTGAME_SCORES  # schermate spurie: da ignorare
    fresh = make_gs()                    # stock tornati a 4
    session = make_bare_session()
    session.console = make_console_stub([None, stale, menu, fresh])
    assert session._wait_fresh_match(timeout=5.0) is True

    session.console = SimpleNamespace(step=lambda: make_gs(p1_stock=1))
    assert session._wait_fresh_match(timeout=0.05) is False


# -- input controller --

def _recording_controller(port, log):
    return SimpleNamespace(
        port=port,
        tilt_analog=lambda b, x, y: log.append(("tilt", port, x, y)),
        press_button=lambda b: log.append(("press", port, b)),
        release_button=lambda b: log.append(("release", port, b)),
        flush=lambda: log.append(("flush", port)),
    )


def test_apply_input_press_and_release():
    log = []
    session = make_bare_session()
    session.controllers = [_recording_controller(1, log), _recording_controller(2, log)]

    session.apply_input(0, button=melee.Button.BUTTON_A, stick_x=1.0, stick_y=0.5, press=True)
    assert log == [("tilt", 1, 1.0, 0.5), ("press", 1, melee.Button.BUTTON_A), ("flush", 1)]

    log.clear()
    session.apply_input(1, button=melee.Button.BUTTON_A, press=False)
    assert log == [("tilt", 2, 0.5, 0.5), ("release", 2, melee.Button.BUTTON_A), ("flush", 2)]

    log.clear()
    session.apply_input(0)  # solo stick, nessun bottone
    assert log == [("tilt", 1, 0.5, 0.5), ("flush", 1)]


def test_press_button_bounds():
    log = []
    session = make_bare_session()
    session.controllers = [_recording_controller(1, log)]
    np.testing.assert_array_equal(session.controller_ports, [1])

    session.press_button(0, melee.Button.BUTTON_A)
    assert log == [("press", 1, melee.Button.BUTTON_A)]
    with pytest.raises(ValueError):
        session.press_button(1, melee.Button.BUTTON_A)
    with pytest.raises(ValueError):
        session.press_button(-1, melee.Button.BUTTON_A)


# -- connessione controller (FIFO) senza open() bloccante --

def test_connect_controller_succeeds_with_reader(tmp_path):
    fifo = tmp_path / "slippibot1"
    os.mkfifo(fifo)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)  # Dolphin finto che apre la pipe
    try:
        session = make_bare_session()
        session.console = SimpleNamespace(controllers=[],
                                          _process=SimpleNamespace(poll=lambda: None))
        controller = SimpleNamespace(pipe_path=str(fifo), port=1, pipe=None)

        session._connect_controller_safe(controller, timeout=2.0)

        assert controller.pipe is not None
        assert session.console.controllers == [controller]
        flags = fcntl.fcntl(controller.pipe.fileno(), fcntl.F_GETFL)
        assert not flags & os.O_NONBLOCK, "la pipe va riportata in modalità bloccante"
        controller.pipe.close()
    finally:
        os.close(reader)


def test_connect_controller_fails_fast_if_dolphin_dead(tmp_path):
    fifo = tmp_path / "slippibot1"
    os.mkfifo(fifo)
    session = make_bare_session()
    session.console = SimpleNamespace(controllers=[], _process=None)  # Dolphin morto

    with pytest.raises(RuntimeError, match="morto"):
        session._connect_controller_safe(SimpleNamespace(pipe_path=str(fifo), port=1, pipe=None),
                                         timeout=2.0)


def test_connect_controller_times_out(tmp_path):
    fifo = tmp_path / "slippibot1"
    os.mkfifo(fifo)
    session = make_bare_session()
    session.console = SimpleNamespace(controllers=[],
                                      _process=SimpleNamespace(poll=lambda: None))  # vivo ma non apre

    with pytest.raises(TimeoutError):
        session._connect_controller_safe(SimpleNamespace(pipe_path=str(fifo), port=1, pipe=None),
                                         timeout=0.3)


# -- pulizia processi: deve colpire SOLO il Dolphin della propria porta --

class FakeProc:
    def __init__(self, name, cmdline, pid=1000):
        self.info = {"name": name, "cmdline": cmdline}
        self.pid = pid
        self.killed = False

    def kill(self):
        self.killed = True


def _fake_libmelee_home(port):
    home = tempfile.mkdtemp(prefix="libmelee_")
    cfg_dir = Path(home) / "Config"
    cfg_dir.mkdir()
    (cfg_dir / "Dolphin.ini").write_text(f"[Core]\nSlippiSpectatorLocalPort = {port}\n")
    return home


def test_kill_stale_dolphins_targets_only_own_port(monkeypatch, tmp_path):
    own_home = _fake_libmelee_home(51442)
    other_home = _fake_libmelee_home(51443)
    try:
        no_ini = tempfile.mkdtemp(prefix="libmelee_")  # home senza Dolphin.ini (boot a metà)

        class DeniedProc(FakeProc):
            def kill(self):
                raise psutil.AccessDenied(self.pid)

        procs = [
            FakeProc("dolphin-emu", ["dolphin-emu", "-u", own_home]),        # stessa porta
            FakeProc("dolphin-emu", ["dolphin-emu", "-u", other_home]),      # altra istanza
            FakeProc("dolphin-emu", ["dolphin-emu"]),                        # senza home
            FakeProc("dolphin-emu", ["dolphin-emu", "-u", str(tmp_path)]),   # non libmelee
            FakeProc("firefox", ["firefox"]),                                # altro processo
            FakeProc("dolphin-emu", ["dolphin-emu", "-u", no_ini]),          # ini mancante
            DeniedProc("dolphin-emu", ["dolphin-emu", "-u", own_home]),      # kill negato
        ]
        monkeypatch.setattr("smash_rl.session.psutil.process_iter", lambda attrs: procs)

        session = make_bare_session(slippi_port=51442)
        session._kill_stale_dolphins()  # l'AccessDenied non deve propagarsi

        assert [p.killed for p in procs[:5]] == [True, False, False, False, False], \
            "va ucciso solo il Dolphin stantio della PROPRIA porta slippi"
    finally:
        shutil.rmtree(own_home, ignore_errors=True)
        shutil.rmtree(other_home, ignore_errors=True)
        shutil.rmtree(no_ini, ignore_errors=True)


def test_force_kill_dolphin_kills_process_tree(monkeypatch):
    killed = []

    class FakePsProcess:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive=False):
            return [SimpleNamespace(kill=lambda: killed.append("figlio"))]

        def kill(self):
            killed.append(self.pid)

    monkeypatch.setattr("smash_rl.session.psutil.Process", FakePsProcess)
    session = make_bare_session()
    session.console = SimpleNamespace(_process=SimpleNamespace(pid=4242))

    session.force_kill_dolphin()
    assert killed == ["figlio", 4242]

    # processo già morto: nessuna eccezione
    def gone(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr("smash_rl.session.psutil.Process", gone)
    session.force_kill_dolphin()


# -- teardown non bloccante --

def _stub_console_for_stop(buffer_items, stop_error=None):
    stopped = []

    def stop():
        stopped.append(True)
        if stop_error:
            raise stop_error

    buf = SimpleNamespace(poll=lambda t: bool(buffer_items),
                          recv_bytes=lambda: buffer_items.pop(0))
    console = SimpleNamespace(_slippstream=SimpleNamespace(_buffer=buf),
                              stop=stop, _process=None)
    return console, stopped


def test_safe_stop_console_drains_buffer_and_stops():
    items = [b"frame1", b"frame2"]
    console, stopped = _stub_console_for_stop(items)
    session = make_bare_session()
    session.console = console
    session.force_kill_dolphin = lambda: None

    session._safe_stop_console()

    assert not items, "il buffer del client enet va svuotato prima dello stop"
    assert stopped == [True]
    assert session.console is None


def test_safe_stop_console_swallows_stop_errors():
    console, stopped = _stub_console_for_stop([], stop_error=RuntimeError("già morto"))
    session = make_bare_session()
    session.console = console
    session.force_kill_dolphin = lambda: None

    session._safe_stop_console()  # non deve sollevare
    assert stopped == [True]
    assert session.console is None


def test_safe_stop_console_tolerates_closed_buffer():
    def poll_closed(t):
        raise EOFError  # il figlio enet ha già chiuso la pipe

    console, stopped = _stub_console_for_stop([])
    console._slippstream._buffer = SimpleNamespace(poll=poll_closed, recv_bytes=lambda: b"")
    session = make_bare_session()
    session.console = console
    session.force_kill_dolphin = lambda: None

    session._safe_stop_console()
    assert stopped == [True]
    assert session.console is None


# -- hard_reset: orchestrazione fail-fast --

def _prepare_hard_reset(session, connect_ok):
    order = []
    session._safe_stop_console = lambda: order.append("safe_stop")
    session._kill_stale_dolphins = lambda: order.append("kill_stale")
    session._ensure_display = lambda: order.append("display")

    def build():
        order.append("build")
        session.console = SimpleNamespace(
            run=lambda iso_path: order.append(("run", iso_path)),
            connect=lambda: connect_ok,
            _process=None,
        )
        session.controllers = [SimpleNamespace(port=1), SimpleNamespace(port=2)]

    session._build_console = build
    session._connect_controller_safe = lambda c: order.append(("controller", c.port))
    return order


def test_hard_reset_success_order():
    session = make_bare_session()
    order = _prepare_hard_reset(session, connect_ok=True)

    assert session.hard_reset() is session

    assert order == ["kill_stale", "display", "build",
                     ("run", str(session.config.iso)), ("controller", 1), ("controller", 2)], \
        "la porta va liberata dai Dolphin stantii PRIMA del nuovo boot"
    assert session._gamestate is None
    assert session._in_game_frames == 0


def test_hard_reset_raises_if_connect_fails():
    session = make_bare_session()
    _prepare_hard_reset(session, connect_ok=False)

    with pytest.raises(RuntimeError, match="connessione a Dolphin fallita"):
        session.hard_reset()


def test_kill_dolphin_uses_pkill(monkeypatch):
    from smash_rl import session as session_module
    runs = []
    monkeypatch.setattr(session_module.subprocess, "run",
                        lambda cmd, **kw: runs.append(cmd))

    session_module.kill_dolphin()
    assert runs == [["pkill", "-f", "dolphin-emu"]]


def test_config_validate_warns_on_rvz(fake_dotenv, capsys):
    rvz = fake_dotenv / "SSBM.rvz"
    rvz.write_text("")
    MeleeConfig(dolphin_exe=fake_dotenv / "dolphin-emu", iso=rvz,
                replay_dir=fake_dotenv / "r").validate()
    assert ".rvz" in capsys.readouterr().out


def test_ensure_display_created_once(monkeypatch):
    created = []

    class FakeDisplay:
        def __init__(self, visible, size):
            created.append(size)

        def start(self):
            pass

    monkeypatch.setattr("smash_rl.session.Display", FakeDisplay)

    session = make_bare_session(headless=True, display_size=(640, 480))
    session._ensure_display()
    session._ensure_display()
    assert created == [(640, 480)], "il display virtuale va creato una volta sola"

    headless_off = make_bare_session(headless=False)
    headless_off._ensure_display()
    assert headless_off.display is None


def test_build_console_reuses_external_console(monkeypatch):
    monkeypatch.setattr("smash_rl.session.melee.Controller",
                        lambda console, port: SimpleNamespace(port=port))
    external = SimpleNamespace()
    session = make_bare_session()
    session._external_console = external

    session._build_console()

    assert session.console is external
    assert [c.port for c in session.controllers] == [1, 2]


def test_connect_controller_raises_on_unexpected_oserror(tmp_path):
    session = make_bare_session()
    session.console = SimpleNamespace(controllers=[], _process=None)
    missing = SimpleNamespace(pipe_path=str(tmp_path / "manca" / "pipe"), port=1, pipe=None)

    with pytest.raises(OSError):  # ENOENT, non ENXIO: errore vero, niente retry
        session._connect_controller_safe(missing, timeout=0.5)


def test_safe_stop_and_force_kill_noop_without_console():
    session = make_bare_session()
    session._safe_stop_console()  # console assente: nessuna eccezione
    session.force_kill_dolphin()
    assert session.console is None


def test_hard_reset_stops_previous_console_first():
    session = make_bare_session()
    order = _prepare_hard_reset(session, connect_ok=True)
    session.console = SimpleNamespace()  # sessione già avviata in precedenza

    session.hard_reset()
    assert order[0] == "safe_stop", \
        "un reboot con console esistente deve prima fermarla in modo non bloccante"


def test_context_manager_boots_and_closes():
    session = make_bare_session()
    _prepare_hard_reset(session, connect_ok=True)
    closes = []
    session.close = lambda: closes.append(True)

    with session as s:
        assert s is session
    assert closes == [True]


def test_close_swallows_display_errors():
    session = make_bare_session()

    def boom():
        raise RuntimeError("Xvfb già chiuso")

    session.display = SimpleNamespace(stop=boom)
    session.close()  # non deve sollevare
    assert session.display is None
    assert session.console is None
