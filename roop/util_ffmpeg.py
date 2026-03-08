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

	commands.extend([ '-map', '0:v', '-map', '1:a', final_video ])
	run_ffmpeg(commands)
