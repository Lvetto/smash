"""
Composizione del video finale: gameplay a sinistra + pannello di diagnostica
a destra, con l'audio del dump riattaccato in coda.

Allineamento video <-> replay: il video dumpato parte dal boot dell'emulatore
(durata variabile), i frame del replay partono da -123. Con -b Dolphin esce a
fine riproduzione, quindi l'ULTIMO frame video corrisponde (a +-2 frame)
all'ultimo frame replay: ancoriamo la mappa alla fine,

    frame_replay(i) = end_frame - (video_frames - 1 - i) + frame_offset

con frame_offset correzione manuale opzionale (verificabile con
write_calibration_video).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from smash_rl.video.dump import PlaybackDumpResult, dump_replay_video
from smash_rl.video.panel import PanelRenderer


def replay_frame_for_video_index(i: int, video_frames: int, end_frame: int,
                                 frame_offset: int = 0) -> int:
    """Frame replay corrispondente al frame video i, ancorando alla fine."""
    return end_frame - (video_frames - 1 - i) + frame_offset


def _open_video(video: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"impossibile aprire il video: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, w, h, fps, n


def _as_dump(dump, records=None) -> PlaybackDumpResult:
    """Accetta PlaybackDumpResult o un path a un mp4 già dumpato."""
    if isinstance(dump, PlaybackDumpResult):
        return dump
    video = Path(dump)
    cap, _, _, fps, n = _open_video(video)
    cap.release()
    if records is None:
        raise ValueError("con un path secco servono i records per stimare end_frame")
    return PlaybackDumpResult(video=video, user_dir=video.parent,
                              start_frame=min(records), end_frame=max(records),
                              video_frames=n, fps=fps)


def _mux_audio(video_only: Path, audio_source: Path, out_path: Path) -> None:
    """Riattacca al video composto la traccia audio del dump (se esiste)."""
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(video_only), "-i", str(audio_source),
         "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy",
         str(out_path)],
        check=True)


def compose_video(dump, records: dict[int, dict], out_path, *,
                  frame_offset: int | None = None,
                  panel_width: int = 480,
                  qval_window: int = 180,
                  max_units_per_column: int = 32,
                  start_frame: int | None = None,
                  end_frame: int | None = None,
                  keep_audio: bool = True,
                  verbose: bool = True) -> Path:
    """
    Affianca al video del gameplay il pannello di diagnostica.

    Args:
        dump: PlaybackDumpResult di dump_replay_video (o path a un mp4, ma in
            quel caso l'allineamento assume che il video finisca sull'ultimo
            frame dei records).
        records: output di extract_frame_records sullo stesso replay.
        frame_offset: correzione manuale all'allineamento (None = 0, cioè solo
            l'ancoraggio automatico alla fine; tararla con write_calibration_video).
        start_frame/end_frame: limita l'output a un intervallo di frame replay
            (clip breve per iterare sul pannello). NB: con un intervallo attivo
            l'audio viene omesso per non desincronizzarlo.
        keep_audio: riattacca l'audio del dump al video finale.
    """
    dump = _as_dump(dump, records)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offset = frame_offset or 0
    sliced = start_frame is not None or end_frame is not None

    cap, w, h, fps, n_video = _open_video(dump.video)
    n_video = min(n_video, dump.video_frames) or dump.video_frames
    renderer = PanelRenderer(h, panel_width, qval_window=qval_window,
                             max_units_per_column=max_units_per_column)

    tmp = Path(tempfile.mkstemp(suffix=".mp4", prefix="smash_compose_")[1])
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w + panel_width, h))
    written = 0
    try:
        for i in range(n_video):
            ok, frame = cap.read()
            if not ok:
                break
            f = replay_frame_for_video_index(i, n_video, dump.end_frame, offset)
            if sliced:
                if start_frame is not None and f < start_frame:
                    continue
                if end_frame is not None and f > end_frame:
                    break
            panel = renderer.render(records.get(f))
            writer.write(np.hstack([frame, panel]))
            written += 1
    finally:
        cap.release()
        writer.release()

    if written == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError("nessun frame scritto: intervallo start/end fuori dal video?")

    if keep_audio and not sliced:
        _mux_audio(tmp, dump.video, out_path)
        tmp.unlink(missing_ok=True)
    else:
        # ricodifica comunque in h264: l'mp4v di opencv è poco compatibile
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(tmp), "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", str(out_path)], check=True)
        tmp.unlink(missing_ok=True)

    if verbose:
        print(f"[replay_video] video annotato: {out_path} ({written} frame)", flush=True)
    return out_path


def write_calibration_video(dump, out_path, *, frame_offset: int = 0,
                            records: dict | None = None) -> Path:
    """
    Copia del video con l'indice di frame replay candidato stampato su ogni
    frame: serve a verificare/tarare frame_offset (es. la scritta 'GO!' deve
    comparire intorno al frame 0).
    """
    dump = _as_dump(dump, records)
    out_path = Path(out_path)
    cap, w, h, fps, n_video = _open_video(dump.video)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    try:
        for i in range(n_video):
            ok, frame = cap.read()
            if not ok:
                break
            f = replay_frame_for_video_index(i, n_video, dump.end_frame, frame_offset)
            beat = "*" * (abs(f) % 60 // 15 + 1)  # battito visivo per contare i frame
            cv2.putText(frame, f"replay frame {f} {beat}", (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    return out_path


def replay_to_annotated_video(slp_path, model_path, out_path, *,
                              obs_name: str = "pos_vel", act_name: str = "a_only",
                              frame_offset: int | None = None,
                              dump_kwargs: dict | None = None,
                              **compose_kwargs) -> Path:
    """
    One-shot: dump del replay + diagnostica del modello + video annotato.

    Per iterare sul pannello senza rifare ogni volta il dump (la parte lenta),
    usare i tre passi separati: dump_replay_video / extract_frame_records /
    compose_video.
    """
    from smash_rl.video.diagnostics import extract_frame_records

    dump = dump_replay_video(slp_path, **(dump_kwargs or {}))
    records = extract_frame_records(slp_path, model_path,
                                    obs_name=obs_name, act_name=act_name)
    return compose_video(dump, records, out_path,
                         frame_offset=frame_offset, **compose_kwargs)
