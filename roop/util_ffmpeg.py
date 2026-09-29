import os
import subprocess as sp
import roop.utilities as util

def run_ffmpeg(commands):
    # Get the path to the local FFmpeg installation
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ffmpeg_paths = [
        os.path.join(project_dir, 'ffmpeg', 'ffmpeg.exe'),
        os.path.join(project_dir, 'ffmpeg', 'ffmpeg-master-latest-win64-gpl', 'bin', 'ffmpeg.exe'),
        'ffmpeg'  # fallback to system PATH
    ]

    ffmpeg_binary = None
    for path in ffmpeg_paths:
        if os.path.exists(path):
            ffmpeg_binary = path
            break
    if ffmpeg_binary is None:
        ffmpeg_binary = 'ffmpeg'

    cmd = [ffmpeg_binary] + commands
    print(f"Ejecutando comando FFmpeg: {' '.join(cmd)}")
    try:
        result = sp.run(cmd, check=True, capture_output=True, text=True)
        print("FFmpeg ejecutado exitosamente")
    except sp.CalledProcessError as e:
        print(f"Error ejecutando FFmpeg: {e}")
        print(f"Salida de error: {e.stderr}")
        raise

def restore_audio(intermediate_video: str, original_video: str, trim_frame_start, trim_frame_end, final_video : str) -> None:
	fps = util.detect_fps(original_video)
	commands = [ '-i', intermediate_video ]
	if trim_frame_start is None and trim_frame_end is None:
		commands.extend([ '-i', original_video, '-c', 'copy' ])
	else:
		if trim_frame_start is not None:
			start_time = trim_frame_start / fps
			commands.extend([ '-ss', format(start_time, ".2f")])
		else:
			commands.extend([ '-ss', '0' ])
		if trim_frame_end is not None:
			end_time = trim_frame_end / fps
			commands.extend([ '-to', format(end_time, ".2f")])
		commands.extend([ '-i', original_video, '-c', 'copy' ])

	# Algunos clips de móvil no tienen audio. El map opcional conserva la
	# pista si existe sin convertir un vídeo correcto en un fallo al finalizar.
	commands.extend([ '-map', '0:v', '-map', '1:a?', final_video ])
	run_ffmpeg(commands)


def encode_face_swap_video(intermediate_video: str, original_video: str, final_video: str,
                           skip_audio: bool = False, start_frame: int = 0,
                           end_frame: int = None, fps: float = None) -> None:
    """Encode a lossless working video once with the project's delivery codec."""
    commands = ['-y', '-i', intermediate_video]
    if not skip_audio:
        if fps and fps > 0 and start_frame:
            commands.extend(['-ss', f'{start_frame / fps:.6f}'])
        if fps and fps > 0 and end_frame is not None:
            duration = max(0.0, (end_frame - start_frame) / fps)
            commands.extend(['-t', f'{duration:.6f}'])
        commands.extend(['-i', original_video])
    commands.extend(['-map', '0:v:0'])
    if not skip_audio:
        commands.extend(['-map', '1:a?'])
    commands.extend([
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '14',
        '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart',
        final_video,
    ])
    run_ffmpeg(commands)
