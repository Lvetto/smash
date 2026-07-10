"""
Integrazione del dump video: avvia la build playback su una clip corta di un
replay vero. Richiede DOLPHIN_PLAYBACK_DIR nel .env (build playback di Slippi)
e un replay in REPLAY_DIR. Eseguire con: pytest -m dolphin -k video
"""
import os

import psutil
import pytest
from dotenv import load_dotenv
from pathlib import Path

from smash_rl.video import dump_replay_video, extract_frame_records, compose_video

pytestmark = pytest.mark.dolphin

load_dotenv()


def _dolphin_count() -> int:
    return sum(1 for p in psutil.process_iter(["name"])
               if "dolphin-emu" in (p.info["name"] or ""))


def _find_replay() -> Path:
    replay_root = Path(os.environ.get("REPLAY_DIR", "replays"))
    for slp in sorted(replay_root.rglob("*.slp")):
        return slp
    pytest.skip(f"nessun replay .slp in {replay_root}")


@pytest.fixture
def playback_available():
    if "DOLPHIN_PLAYBACK_DIR" not in os.environ:
        pytest.skip("DOLPHIN_PLAYBACK_DIR non configurata nel .env")


def test_dump_short_clip(tmp_path, playback_available):
    slp = _find_replay()
    before = _dolphin_count()

    dump = dump_replay_video(slp, tmp_path / "clip.mp4",
                             start_frame=-123, end_frame=600)  # ~12s di gioco

    assert dump.video.is_file()
    assert dump.video_frames > 0
    assert dump.end_frame <= 600
    assert _dolphin_count() == before          # nessun processo appeso
    assert not dump.user_dir.exists()          # user dir temporanea rimossa


def test_end_to_end_annotated_clip(tmp_path, playback_available):
    slp = _find_replay()
    weights = sorted(Path("weights").glob("*.zip"))
    if not weights:
        pytest.skip("nessun checkpoint in weights/")

    dump = dump_replay_video(slp, tmp_path / "clip.mp4",
                             start_frame=-123, end_frame=600)
    records = extract_frame_records(slp, weights[0])
    out = compose_video(dump, records, tmp_path / "annotated.mp4")
    assert out.is_file() and out.stat().st_size > 0
