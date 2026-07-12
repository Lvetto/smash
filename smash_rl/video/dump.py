"""
Dump video di un replay .slp tramite la build *playback* di Slippi.

La build playback (diversa da quella netplay e da quella EXI del training;
path atteso in .env come DOLPHIN_PLAYBACK_DIR) si pilota interamente da CLI:

    dolphin-emu -i comm.json -b -e SSBM.iso -u <user_dir> --hide-seekbar --cout

dove comm.json indica il replay da riprodurre. Il dump dei frame e dell'audio
si abilita via INI nella user dir; con -b Dolphin esce da solo a fine
riproduzione. I file grezzi (framedump0.avi + dspdump.wav) vengono poi uniti
in un mp4 con ffmpeg.

Non usiamo melee.Console: Console.run non permette i flag extra (-i, -b,
--cout) e sovrascriverebbe i gecko codes con quelli del bot, dannosi per il
playback. Qui non serve nemmeno connettersi al processo: solo lanciarlo,
aspettarlo e raccogliere i file.
"""
from __future__ import annotations

import configparser
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import os

import psutil
from dotenv import load_dotenv
from pyvirtualdisplay import Display

from smash_rl.session import MeleeConfig
from smash_rl.video.diagnostics import slp_frame_bounds

FIRST_REPLAY_FRAME = -123   # i replay Slippi partono dal countdown pre-partita

# righe informative stampate dal playback con --cout, es. "[CURRENT_FRAME] 1042"
_COUT_RE = re.compile(r"\[([A-Z_]+)\](?:\s+(-?\d+))?")


@dataclass
class PlaybackDumpResult:
    """Esito del dump: video muxato + i dati che servono per l'allineamento."""
    video: Path         # mp4 finale (video + audio se disponibile)
    user_dir: Path      # home Dolphin temporanea (rimossa se cleanup, tenuta per debug altrimenti)
    start_frame: int    # frame replay a inizio playback
    end_frame: int      # ultimo frame replay riprodotto
    video_frames: int   # numero di frame del video dumpato
    fps: float


def parse_cout_line(line: str):
    """'[CURRENT_FRAME] 42' -> ('CURRENT_FRAME', 42); None se non è una riga --cout."""
    m = _COUT_RE.search(line)
    if m is None:
        return None
    key, val = m.group(1), m.group(2)
    return key, (int(val) if val is not None else None)


def write_comm_file(path: Path, slp_path: Path, start_frame: int, end_frame: int) -> Path:
    """Scrive il comm file JSON che dice al playback quale replay riprodurre."""
    comm = {
        "mode": "normal",
        "replay": str(slp_path.resolve()),
        "startFrame": start_frame,
        "endFrame": end_frame,
        "isRealTimeMode": False,
        "shouldResync": True,
        "rollbackDisplayMethod": "off",
        "commandId": uuid.uuid4().hex,
    }
    path.write_text(json.dumps(comm, indent=2))
    return path


def write_dump_inis(user_dir: Path, *, bitrate_kbps: int = 15000,
                    efb_scale: int = 2) -> None:
    """
    Configura la user dir di Dolphin per il dump: frame in Dump/Frames/*.avi,
    audio in Dump/Audio/dspdump.wav. (Chiavi: cfr. DumpConfig in melee/console.py.)
    """
    config_dir = user_dir / "Config"
    config_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, sections: dict[str, dict[str, str]]) -> None:
        ini_path = config_dir / name
        cfg = configparser.ConfigParser()
        cfg.optionxform = str  # preserva le maiuscole delle chiavi
        if ini_path.is_file():
            cfg.read(ini_path)
        for section, options in sections.items():
            if not cfg.has_section(section):
                cfg.add_section(section)
            for k, v in options.items():
                cfg.set(section, k, v)
        with open(ini_path, "w") as f:
            cfg.write(f)

    _write("Dolphin.ini", {
        "Movie": {"DumpFrames": "True", "DumpFramesSilent": "True"},
        # DumpAudioSilent evita il popup "sovrascrivere?" che bloccherebbe l'headless
        "DSP": {"DumpAudio": "True", "DumpAudioSilent": "True", "Backend": "Pulse"},
        "Display": {"Fullscreen": "False"},
        "Core": {"EmulationSpeed": "1.0"},
        "Analytics": {"Enabled": "False", "PermissionAsked": "True"},
    })
    _write("GFX.ini", {
        "Settings": {
            "InternalResolutionFrameDumps": "True",
            "BitrateKbps": str(bitrate_kbps),
            "EFBScale": str(efb_scale),
        },
    })


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """SIGKILL al processo e ai figli (stesso approccio di MeleeSession.force_kill_dolphin)."""
    try:
        p = psutil.Process(proc.pid)
        for child in p.children(recursive=True):
            child.kill()
        p.kill()
    except psutil.NoSuchProcess:
        pass


def _count_video_frames(video: Path) -> tuple[int, float]:
    """(n_frame, fps) del video, via opencv."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    finally:
        cap.release()
    return n, fps


def dump_replay_video(slp_path, out_path=None, *,
                      start_frame: int | None = None,
                      end_frame: int | None = None,
                      dotenv_path: str | None = None,
                      headless: bool = True,
                      display_size: tuple[int, int] = (960, 720),
                      timeout_s: float | None = None,
                      end_grace_s: float = 5.0,
                      keep_user_dir: bool = False,
                      template_user_dir: Path | None = None,
                      bitrate_kbps: int = 15000,
                      verbose: bool = True) -> PlaybackDumpResult:
    """
    Riproduce il replay con la build playback e ritorna un mp4 del gameplay.

    Richiede in .env: DOLPHIN_PLAYBACK_DIR (build playback di Slippi, installala
    dal Slippi Launcher o dalle release playback di project-slippi) e
    SMBM_ISO_PATH. Richiede ffmpeg nel PATH.

    Args:
        start_frame/end_frame: intervallo di frame replay da riprodurre
            (default: tutta la partita). Utile per clip corte di prova.
        timeout_s: watchdog complessivo (default: 60 + 1.5x la durata attesa).
        end_grace_s: rete di sicurezza. Con -b Dolphin esce da solo a fine
            playback (segnale primario: EOF sullo stdout). Se non lo fa,
            end_grace_s sono i secondi concessi per chiudere il dump dopo
            aver raggiunto il frame finale (da CURRENT_FRAME), prima del kill.
        keep_user_dir: tiene la home Dolphin temporanea (per debug).
        template_user_dir: user dir esistente da cui copiare la config di
            partenza (escape hatch se il playback desincronizza con la home vuota).
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg non trovato nel PATH: installalo (es. apt install ffmpeg)")

    slp_path = Path(slp_path)
    if not slp_path.is_file():
        raise FileNotFoundError(f"replay non trovato: {slp_path}")

    load_dotenv(dotenv_path)
    if "DOLPHIN_PLAYBACK_DIR" not in os.environ:
        raise KeyError("DOLPHIN_PLAYBACK_DIR mancante nel .env: serve la build "
                       "playback di Slippi (Slippi Launcher o release playback di "
                       "project-slippi), NON quella netplay o EXI")
    exe = MeleeConfig._resolve_dolphin_exe(Path(os.environ["DOLPHIN_PLAYBACK_DIR"]))
    iso = Path(os.environ["SMBM_ISO_PATH"])
    if not exe.is_file():
        raise FileNotFoundError(f"eseguibile playback non trovato: {exe}")
    if not iso.is_file():
        raise FileNotFoundError(f"ISO non trovata: {iso}")

    first, last = slp_frame_bounds(slp_path)
    start_frame = first if start_frame is None else max(start_frame, first)
    end_frame = last if end_frame is None else min(end_frame, last)
    n_frames = end_frame - start_frame + 1
    if timeout_s is None:
        timeout_s = 60.0 + 1.5 * n_frames / 60.0

    out_path = Path(out_path) if out_path is not None else slp_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # prefisso diverso da libmelee_*: MeleeSession._kill_stale_dolphins non deve toccarla
    user_dir = Path(tempfile.mkdtemp(prefix="smash_video_"))
    if template_user_dir is not None:
        shutil.copytree(template_user_dir, user_dir, dirs_exist_ok=True)
    comm_path = user_dir / "comm.json"
    write_comm_file(comm_path, slp_path, start_frame, end_frame)
    write_dump_inis(user_dir, bitrate_kbps=bitrate_kbps)

    def log(msg):
        if verbose:
            print(f"[replay_video] {msg}", flush=True)

    display = None
    proc = None
    cout_state: dict[str, int | None] = {}
    try:
        if headless:
            display = Display(visible=0, size=display_size)
            display.start()

        cmd = [str(exe), "-i", str(comm_path), "-b", "-e", str(iso),
               "-u", str(user_dir), "--hide-seekbar", "--cout"]
        log(f"lancio playback: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)

        # watchdog: -b dovrebbe far uscire Dolphin da solo, ma non fidiamoci
        watchdog = threading.Timer(timeout_s, _kill_process_tree, args=(proc,))
        watchdog.daemon = True
        watchdog.start()
        grace_timer = None
        # GAME_END_FRAME/PLAYBACK_END_FRAME sono metadati emessi all'avvio del
        # playback: dicono DOVE finirà la riproduzione, non che è già finita.
        # Il progresso reale arriva via CURRENT_FRAME; il segnale primario di
        # fine resta comunque l'uscita spontanea di Dolphin (EOF sullo stdout,
        # grazie a -b) letta dal for sotto.
        target_end = end_frame
        t0 = time.time()
        try:
            for line in proc.stdout:
                parsed = parse_cout_line(line)
                if parsed is None:
                    continue
                key, val = parsed
                cout_state[key] = val
                if key == "NO_GAME":
                    _kill_process_tree(proc)
                    raise RuntimeError(f"il playback non ha trovato la partita nel replay {slp_path}")
                if key == "PLAYBACK_END_FRAME" and val is not None:
                    target_end = val
                if (key == "CURRENT_FRAME" and val is not None
                        and val >= target_end and grace_timer is None):
                    # raggiunto il frame finale: rete di sicurezza nel caso
                    # -b non chiuda Dolphin da solo dopo aver finito il dump
                    grace_timer = threading.Timer(end_grace_s, _kill_process_tree, args=(proc,))
                    grace_timer.daemon = True
                    grace_timer.start()
            proc.wait()
        finally:
            watchdog.cancel()
            if grace_timer is not None:
                grace_timer.cancel()
            if proc.poll() is None:
                _kill_process_tree(proc)
                proc.wait()
        log(f"playback terminato in {time.time() - t0:.0f}s (cout: {cout_state or 'nessun messaggio'})")

        # raccolta dei dump
        frame_dumps = sorted((user_dir / "Dump" / "Frames").glob("framedump*.avi"))
        if not frame_dumps:
            raise RuntimeError(
                f"nessun framedump prodotto in {user_dir}/Dump/Frames: controlla che "
                f"DOLPHIN_PLAYBACK_DIR punti alla build playback (usa keep_user_dir=True per il debug)")
        audio_dump = user_dir / "Dump" / "Audio" / "dspdump.wav"

        # mux in mp4 (concat se il dump è spezzato in più file, succede al cambio risoluzione)
        ffmpeg_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        if len(frame_dumps) == 1:
            ffmpeg_cmd += ["-i", str(frame_dumps[0])]
        else:
            concat_list = user_dir / "framedumps.txt"
            concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in frame_dumps))
            ffmpeg_cmd += ["-f", "concat", "-safe", "0", "-i", str(concat_list)]
        if audio_dump.is_file():
            ffmpeg_cmd += ["-i", str(audio_dump), "-c:a", "aac", "-shortest"]
        else:
            log("ATTENZIONE: dspdump.wav mancante, il video sarà senza audio")
        ffmpeg_cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out_path)]
        log("mux ffmpeg...")
        subprocess.run(ffmpeg_cmd, check=True)

        video_frames, fps = _count_video_frames(out_path)
        log(f"fatto: {out_path} ({video_frames} frame @ {fps:.1f}fps, "
            f"replay {start_frame}..{end_frame})")

        # se il playback ha detto dove ha smesso di dumpare, fidiamoci di lui per
        # l'allineamento: PLAYBACK_END_FRAME è l'ultimo frame riprodotto (tiene
        # conto del taglio a end_frame), GAME_END_FRAME è la fine naturale della
        # partita (irrilevante quando tagliamo una clip).
        observed_end = (cout_state.get("PLAYBACK_END_FRAME")
                        or cout_state.get("GAME_END_FRAME")
                        or cout_state.get("CURRENT_FRAME"))
        if observed_end is not None:
            end_frame = min(end_frame, observed_end)

        return PlaybackDumpResult(video=out_path, user_dir=user_dir,
                                  start_frame=start_frame, end_frame=end_frame,
                                  video_frames=video_frames, fps=fps)
    finally:
        if display is not None:
            display.stop()
        if not keep_user_dir:
            shutil.rmtree(user_dir, ignore_errors=True)
