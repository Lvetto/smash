"""
Unit test per smash_rl/video (niente Dolphin: il dump vero è coperto dal
test di integrazione marcato 'dolphin' quando la build playback è installata).
"""
import configparser
import json
from pathlib import Path

import numpy as np
import pytest

from smash_rl.specs.actions import ACT_LABELS, ACT_LAYOUTS, ACT_SPECS
from smash_rl.specs.observations import OBS_FEATURE_NAMES, OBS_SPECS
from smash_rl.tests.helpers import make_gs
from smash_rl.video.compose import replay_frame_for_video_index, specs_for_replay
from smash_rl.video.diagnostics import ReplaySessionShim
from smash_rl.video.dump import parse_cout_line, write_comm_file, write_dump_inis
from smash_rl.video.panel import PanelRenderer
from smash_rl.specs.context import Ctx


# -- dump.py: comm file e INI --

def test_write_comm_file(tmp_path):
    slp = tmp_path / "game.slp"
    slp.write_bytes(b"")
    comm = write_comm_file(tmp_path / "comm.json", slp, -123, 3588)
    data = json.loads(comm.read_text())
    assert data["replay"] == str(slp.resolve())
    assert data["mode"] == "normal"
    assert data["startFrame"] == -123
    assert data["endFrame"] == 3588
    assert data["isRealTimeMode"] is False
    assert data["commandId"]


def test_write_dump_inis(tmp_path):
    write_dump_inis(tmp_path, bitrate_kbps=12345, efb_scale=3)

    dolphin = configparser.ConfigParser()
    dolphin.optionxform = str
    dolphin.read(tmp_path / "Config" / "Dolphin.ini")
    assert dolphin["Movie"]["DumpFrames"] == "True"
    assert dolphin["Movie"]["DumpFramesSilent"] == "True"
    assert dolphin["DSP"]["DumpAudio"] == "True"

    gfx = configparser.ConfigParser()
    gfx.optionxform = str
    gfx.read(tmp_path / "Config" / "GFX.ini")
    assert gfx["Settings"]["InternalResolutionFrameDumps"] == "True"
    assert gfx["Settings"]["BitrateKbps"] == "12345"
    assert gfx["Settings"]["EFBScale"] == "3"


def test_write_dump_inis_preserves_existing(tmp_path):
    (tmp_path / "Config").mkdir()
    (tmp_path / "Config" / "Dolphin.ini").write_text("[Core]\nSlippiReplayDir = /x\n")
    write_dump_inis(tmp_path)
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(tmp_path / "Config" / "Dolphin.ini")
    assert cfg["Core"]["SlippiReplayDir"] == "/x"       # non cancellata
    assert cfg["Movie"]["DumpFrames"] == "True"


def test_parse_cout_line():
    assert parse_cout_line("[CURRENT_FRAME] 42") == ("CURRENT_FRAME", 42)
    assert parse_cout_line("[GAME_END_FRAME] -5") == ("GAME_END_FRAME", -5)
    assert parse_cout_line("[NO_GAME]") == ("NO_GAME", None)
    assert parse_cout_line("qualsiasi altra riga di log") is None


# -- diagnostics.py: shim della sessione --

def test_replay_session_shim_matches_manual_features():
    gs = make_gs(p1_pos=(17.0, 5.0), p2_pos=(-34.0, 10.0),
                 p1_vel=(1.0, -2.0, 0.5, 0.0), distance=42.0)
    shim = ReplaySessionShim(n_players=2)
    shim.set_gamestate(gs)
    ctx = Ctx(agent_port=1, opp_port=2, session=shim)

    obs = OBS_SPECS["pos_vel"][1](gs, ctx)
    assert obs.shape == (12,)
    np.testing.assert_allclose(obs[0], 17.0 / 85.0)      # x agente / STAGE_X_MAX
    np.testing.assert_allclose(obs[1], 5.0 / 50.0)       # y agente / STAGE_Y_MAX
    np.testing.assert_allclose(obs[2], 1.0 / 5.0)        # vx_self / VEL_NORM
    np.testing.assert_allclose(obs[6], -34.0 / 85.0)     # x avversario
    assert shim.distance == 42.0


# -- compose.py: allineamento --

@pytest.mark.parametrize("i,n_video,end_frame,offset,expected", [
    (399, 400, 3588, 0, 3588),   # ultimo frame video -> ultimo frame replay
    (398, 400, 3588, 0, 3587),
    (0, 400, 3588, 0, 3189),
    (399, 400, 3588, -2, 3586),  # correzione manuale
])
def test_replay_frame_for_video_index(i, n_video, end_frame, offset, expected):
    assert replay_frame_for_video_index(i, n_video, end_frame, offset) == expected


# -- panel.py --

def _fake_record(action=3, n_actions=18, obs_dim=12, button=None):
    q = np.linspace(0.0, 1.0, n_actions).astype(np.float32)
    q[action] = 2.0
    return {
        "obs": np.random.uniform(-1, 1, obs_dim).astype(np.float32),
        "q_values": q,
        "action": action,
        "stick": (0.5, 1.0),
        "button": button,
        "activations": [np.random.rand(64).astype(np.float32) for _ in range(2)],
        "percents": (12.0, 96.5),
        "stocks": (4, 3),
        "positions": np.zeros((2, 2)),
        "distance": 10.0,
    }


def test_panel_renderer_shape_and_no_data():
    r = PanelRenderer(480, 320)      # senza etichette: fallback (barplot, neuroni anonimi)
    img = r.render(_fake_record())
    assert img.shape == (480, 320, 3)
    assert img.dtype == np.uint8
    assert img.any()                      # non tutto nero
    img_nd = r.render(None)               # frame senza dati: non deve esplodere
    assert img_nd.shape == (480, 320, 3)


def test_panel_renderer_labeled_full():
    """Pannello etichettato con la spec full (54 azioni = griglia 9x6, obs 30)."""
    import melee
    r = PanelRenderer(720, 480,
                      obs_labels=OBS_FEATURE_NAMES["full_obs"],
                      act_labels=ACT_LABELS["full"],
                      act_layout=ACT_LAYOUTS["full"])
    rec = _fake_record(action=20, n_actions=ACT_SPECS["full"][0].n, obs_dim=30,
                       button=melee.Button.BUTTON_B)
    img = r.render(rec)
    assert img.shape == (720, 480, 3)
    assert img.any()


def test_panel_renderer_labeled_a_only():
    """Pannello etichettato con la spec a_only (18 azioni = griglia 9x2)."""
    r = PanelRenderer(720, 480,
                      obs_labels=OBS_FEATURE_NAMES["pos_vel"],
                      act_labels=ACT_LABELS["a_only"],
                      act_layout=ACT_LAYOUTS["a_only"])
    img = r.render(_fake_record(action=5, n_actions=ACT_SPECS["a_only"][0].n, obs_dim=12))
    assert img.shape == (720, 480, 3)
    assert img.any()


def test_panel_renderer_100_frames_fast():
    import time
    r = PanelRenderer(480, 480)
    rec = _fake_record()
    t0 = time.time()
    for _ in range(100):
        r.render(rec)
    assert time.time() - t0 < 1.0


# -- coerenza etichette <-> spazi delle spec --

@pytest.mark.parametrize("name", list(OBS_FEATURE_NAMES))
def test_obs_feature_names_match_shape(name):
    space, _ = OBS_SPECS[name]
    assert len(OBS_FEATURE_NAMES[name]) == space.shape[0]


@pytest.mark.parametrize("name", list(ACT_LABELS))
def test_act_labels_match_space(name):
    space, _ = ACT_SPECS[name]
    assert len(ACT_LABELS[name]) == space.n
    layout = ACT_LAYOUTS[name]
    assert len(layout["stick_labels"]) * len(layout["button_labels"]) == space.n


def test_specs_for_replay_reads_config(tmp_path):
    run = "run_x"
    (tmp_path / "runs" / run).mkdir(parents=True)
    (tmp_path / "runs" / run / "config.json").write_text(json.dumps({
        "env_kwargs": {"observation_function": "full_obs", "action_function": "full"}
    }))
    slp = tmp_path / "replays" / run / "instance_3" / "Game.slp"
    slp.parent.mkdir(parents=True)
    slp.write_bytes(b"")
    obs_name, act_name = specs_for_replay(slp, runs_dir=tmp_path / "runs")
    assert (obs_name, act_name) == ("full_obs", "full")


def test_specs_for_replay_missing_config(tmp_path):
    slp = tmp_path / "replays" / "sconosciuta" / "instance_0" / "Game.slp"
    slp.parent.mkdir(parents=True)
    slp.write_bytes(b"")
    assert specs_for_replay(slp, runs_dir=tmp_path / "runs") == (None, None)
