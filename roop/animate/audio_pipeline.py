#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audio para Animate Image: ambiente (MMAudio) + voz español (XTTS)."""

import os
import re
import subprocess
import tempfile
from typing import Callable, Optional, Tuple

from roop.utils import get_ffmpeg_path


def extract_speech_text(prompt: str) -> str:
    """Extrae texto hablable: comillas primero, si no el prompt limpio."""
    prompt = (prompt or "").strip()
    if not prompt:
        return ""
    quoted = re.findall(r'["\']([^"\']{2,})["\']', prompt)
    if quoted:
        return quoted[0].strip()
    return prompt.strip()


def mux_audio_track(video_path: str, audio_path: str, *, mix_existing: bool = False) -> Optional[str]:
    ffmpeg = get_ffmpeg_path()
    out_path = video_path.replace(".mp4", "_av.mp4")
    backup = video_path.replace(".mp4", "_preaudio.mp4")
    try:
        if os.path.exists(backup):
            os.remove(backup)
        os.rename(video_path, backup)
        if mix_existing:
            cmd = [
                ffmpeg, "-hide_banner", "-y",
                "-i", backup,
                "-i", audio_path,
                "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=shortest:dropout_transition=2[aout]",
                "-map", "0:v:0",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                out_path,
            ]
        else:
            cmd = [
                ffmpeg, "-hide_banner", "-y",
                "-i", backup,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                out_path,
            ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[AnimateAudio] FFmpeg mux: {result.stderr[-400:]}")
            if not os.path.exists(video_path):
                os.rename(backup, video_path)
            return None
        if os.path.exists(backup):
            os.remove(backup)
        if os.path.exists(video_path):
            os.remove(video_path)
        os.rename(out_path, video_path)
        return video_path
    except Exception as e:
        print(f"[AnimateAudio] Mux error: {e}")
        if not os.path.exists(video_path) and os.path.exists(backup):
            os.rename(backup, video_path)
        return None


def add_spanish_speech(
    video_path: str,
    speech_text: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, str]:
    if not speech_text or not os.path.exists(video_path):
        return video_path, ""

    if progress_callback:
        progress_callback("Generando voz en español (XTTS)...")

    try:
        from roop.audio_generator import generate_audio
    except ImportError:
        return video_path, "voz omitida (TTS no instalado)"

    wav_path = os.path.join(tempfile.gettempdir(), f"animate_tts_{int(os.path.getmtime(video_path))}.wav")
    try:
        out = generate_audio(speech_text, "Español", output_path=wav_path)
        if not out or not os.path.isfile(out):
            return video_path, "voz omitida (XTTS falló)"
        has_audio = False
        try:
            probe = subprocess.run(
                [get_ffmpeg_path(), "-hide_banner", "-i", video_path],
                capture_output=True, text=True, timeout=30,
            )
            has_audio = "Audio:" in (probe.stderr or "")
        except Exception:
            pass
        merged = mux_audio_track(video_path, out, mix_existing=has_audio)
        if merged:
            print(f"[AnimateAudio] Voz español añadida: {speech_text[:80]}...")
            return merged, "voz español"
        return video_path, "voz omitida (mux falló)"
    except Exception as e:
        print(f"[AnimateAudio] XTTS error: {e}")
        return video_path, f"voz omitida ({e})"


def apply_animate_audio(
    video_path: str,
    *,
    user_prompt: str,
    motion_prompt: str,
    speech_intensity: float = 0.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, str]:
    """Ambiente automático + voz español si el análisis detecta diálogo."""
    notes = []
    out = video_path

    from roop.animate.mmaudio_client import add_audio_to_video, get_status_message, is_available

    want_speech = speech_intensity >= 0.10
    ambient_prompt = user_prompt or motion_prompt

    if is_available(check_comfy=True):
        if progress_callback:
            progress_callback("Generando sonido ambiente (MMAudio)...")
        out, mm_msg = add_audio_to_video(
            out,
            sound_prompt=ambient_prompt,
            motion_prompt=motion_prompt,
            progress_callback=progress_callback,
        )
        if mm_msg and "omitido" not in mm_msg.lower():
            notes.append("ambiente")
    else:
        notes.append(get_status_message(check_comfy=True))

    if want_speech:
        speech_text = extract_speech_text(user_prompt or motion_prompt)
        if speech_text:
            out, speech_note = add_spanish_speech(out, speech_text, progress_callback=progress_callback)
            if speech_note and "omitida" not in speech_note:
                notes.append(speech_note)

    return out, " + ".join(n for n in notes if n) or "sin audio"