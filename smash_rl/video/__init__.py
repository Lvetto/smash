"""
Generazione di video annotati dai replay .slp: gameplay (dump dalla build
playback di Slippi) + pannello laterale con la diagnostica del modello
(azione scelta, Q-values, attivazioni della rete).

Uso tipico da notebook:

    from smash_rl.video import replay_to_annotated_video
    replay_to_annotated_video("replays/.../Game_x.slp", "weights/model.zip", "out.mp4")

Oppure a passi separati (utile per iterare sul pannello senza rifare il dump):

    from smash_rl.video import dump_replay_video, extract_frame_records, compose_video
    dump = dump_replay_video("replays/.../Game_x.slp")
    recs = extract_frame_records("replays/.../Game_x.slp", "weights/model.zip")
    compose_video(dump, recs, "out.mp4")
"""
from smash_rl.video.diagnostics import extract_frame_records, slp_frame_bounds
from smash_rl.video.dump import PlaybackDumpResult, dump_replay_video
from smash_rl.video.compose import (compose_video, replay_to_annotated_video,
                                    specs_for_replay, write_calibration_video)

__all__ = [
    "PlaybackDumpResult",
    "dump_replay_video",
    "extract_frame_records",
    "slp_frame_bounds",
    "compose_video",
    "write_calibration_video",
    "replay_to_annotated_video",
    "specs_for_replay",
]
