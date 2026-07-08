"""
Test del loop di gioco umano-vs-agente (smash_rl/play.py): cadenza delle decisioni,
uscita a fine match e estrazione del template del gamepad. Nessun Dolphin.
"""
import configparser
from types import SimpleNamespace

import melee
import numpy as np
import pytest
from gymnasium import spaces

from smash_rl.play import _extract_pad_section, _print_result, main, run_match
from smash_rl.specs.actions import ACT_SPECS
from smash_rl.specs.context import Ctx
from smash_rl.tests.helpers import TEST_OBS_SHAPE, FakeSession, make_gs


def _make_match(frames):
    session = FakeSession(frames)
    ctx = Ctx(agent_port=1, opp_port=2, session=session)
    from smash_rl.specs.observations import OBS_SPECS
    _, build_obs = OBS_SPECS["test_minimal"]
    _, decode_act = ACT_SPECS["a_only"]
    return session, ctx, build_obs, decode_act


def test_run_match_decides_every_frame_skip_and_stops_on_match_over():
    # 2 frame normali, poi l'umano perde l'ultimo stock
    session, ctx, build_obs, decode = _make_match(
        [make_gs(frame=1), make_gs(frame=2), make_gs(frame=3, p2_stock=0)])
    seen_obs = []

    def predict(obs):
        seen_obs.append(obs)
        return 1   # stick neutro + BUTTON_A in "a_only"

    gs = run_match(session, predict, ctx, build_obs, decode, frame_skip=3)

    assert gs.frame == 3
    # una sola decisione (k=0): i frame 2 e 3 cadono dentro la finestra di skip
    assert len(seen_obs) == 1
    assert seen_obs[0].shape == TEST_OBS_SHAPE
    # input solo sulla porta agente, press sul primo frame e release sul secondo;
    # nessun input sul frame di match_over (si esce prima di applicarlo)
    assert [i["player_idx"] for i in session.inputs] == [0, 0]
    assert [i["press"] for i in session.inputs] == [True, False]
    assert all(i["button"] == melee.Button.BUTTON_A for i in session.inputs)


def test_run_match_new_decision_after_frame_skip():
    frames = [make_gs(frame=f) for f in range(1, 5)] + [make_gs(frame=5, p2_stock=0)]
    session, ctx, build_obs, decode = _make_match(frames)
    calls = []
    run_match(session, lambda obs: calls.append(1) or 0, ctx, build_obs, decode,
              frame_skip=2)
    # 4 frame giocati con skip 2 = decisioni a k=0 e k=2
    assert len(calls) == 2
    assert [i["press"] for i in session.inputs] == [True, False, True, False]


def test_run_match_returns_when_leaving_in_game():
    menu = make_gs()
    menu.menu_state = melee.Menu.CHARACTER_SELECT
    session, ctx, build_obs, decode = _make_match([menu])

    gs = run_match(session, lambda obs: 0, ctx, build_obs, decode)

    assert gs is menu
    assert session.inputs == [], "fuori dall'IN_GAME non va inviato nessun input"


def test_run_match_skips_none_frames():
    # frame None in mezzo (lag/rollback): niente input, si aspetta il successivo
    session, ctx, build_obs, decode = _make_match(
        [make_gs(frame=1), None, make_gs(frame=2, p2_stock=0)])
    run_match(session, lambda obs: 0, ctx, build_obs, decode, frame_skip=3)
    assert len(session.inputs) == 1


def test_run_match_resyncs_old_stocks():
    # la partita precedente è finita 0-4: senza risincronizzare old_stocks il
    # match_over scatterebbe subito sul primo frame (stock "risaliti" a 4)
    session, ctx, build_obs, decode = _make_match(
        [make_gs(frame=2), make_gs(frame=3, p2_stock=0)])
    session._gamestate = make_gs(frame=1)
    session.old_stocks = [0, 4]

    gs = run_match(session, lambda obs: 0, ctx, build_obs, decode)

    assert gs.frame == 3, "il match deve finire per gli stock a 0, non per il reset fantasma"


# -- estrazione del template del pad dalla config generata dalla GUI di Dolphin --

def test_extract_pad_section(tmp_path):
    src = tmp_path / "GCPadNew.ini"
    src.write_text("[GCPad1]\nDevice = evdev/0/PadFinto\nButtons/A = SOUTH\n"
                   "Main Stick/Up = Axis 1-\n[GCPad2]\nDevice = altro\n")
    out = tmp_path / "template.ini"

    _extract_pad_section(src, out)

    parsed = configparser.ConfigParser()
    parsed.optionxform = str
    parsed.read(out)
    assert parsed.sections() == ["GCPad"]
    assert parsed["GCPad"]["Device"] == "evdev/0/PadFinto"
    assert parsed["GCPad"]["Buttons/A"] == "SOUTH"
    assert parsed["GCPad"]["Main Stick/Up"] == "Axis 1-"


def test_extract_pad_section_requires_mapped_port1(tmp_path):
    out = tmp_path / "template.ini"

    empty = tmp_path / "vuoto.ini"
    empty.write_text("")
    with pytest.raises(RuntimeError, match="mapping"):
        _extract_pad_section(empty, out)

    no_device = tmp_path / "senza_device.ini"
    no_device.write_text("[GCPad1]\nButtons/A = SOUTH\n")
    with pytest.raises(RuntimeError, match="mapping"):
        _extract_pad_section(no_device, out)

    assert not out.exists()


# -- esito della partita e guardie della CLI --

def test_print_result_all_outcomes(capsys):
    session = FakeSession([])

    session._gamestate = make_gs(p1_stock=2, p2_stock=0)
    _print_result(session, melee.Character.FOX)
    assert "L'agente (FOX) ha vinto" in capsys.readouterr().out

    session._gamestate = make_gs(p1_stock=0, p2_stock=3)
    _print_result(session, melee.Character.FOX)
    assert "Hai vinto" in capsys.readouterr().out

    session._gamestate = make_gs(p1_stock=4, p2_stock=4)
    _print_result(session, melee.Character.FOX)
    assert "resettato" in capsys.readouterr().out

    session._gamestate = None   # partita mai iniziata/crash
    _print_result(session, melee.Character.FOX)
    assert "interrotta" in capsys.readouterr().out


def test_main_rejects_checkpoint_with_wrong_specs(monkeypatch):
    # sbagliare --obs/--act farebbe giocare l'agente su input senza senso: meglio fallire subito
    fake_model = SimpleNamespace(observation_space=spaces.Box(-1.0, 1.0, (99,), np.float32),
                                 action_space=spaces.Discrete(3))
    monkeypatch.setattr("smash_rl.play.DQN",
                        SimpleNamespace(load=lambda *a, **k: fake_model))

    with pytest.raises(SystemExit, match="obs"):
        main(["modello.zip", "--obs", "pos_vel"])

    fake_model.observation_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
    with pytest.raises(SystemExit, match="azioni"):
        main(["modello.zip", "--obs", "pos_vel", "--act", "a_only"])


def test_main_dispatches_configure_pad(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr("smash_rl.play.configure_pad", lambda out: called.append(out))

    main(["--configure-pad", "--pad-config", str(tmp_path / "pad.ini")])

    assert called == [tmp_path / "pad.ini"]
