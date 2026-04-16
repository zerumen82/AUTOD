import os
import shutil
import roop.globals


def run_command(command, mode="silent"):
    if mode == "debug":
        return os.system(command)
    return os.popen(command).read()


def get_ffmpeg_path():
    """Get the path to FFmpeg executable, checking local directory first"""
    # Try local FFmpeg first
    local_ffmpeg = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ffmpeg', 'ffmpeg.exe')
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg

    # Try alternative local paths
    alt_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ffmpeg', 'ffmpeg-master-latest-win64-gpl', 'bin', 'ffmpeg.exe'),
    ]
    for path in alt_paths:
        if os.path.exists(path):
            return path

    # Fall back to system PATH
    return 'ffmpeg'


def get_ffprobe_path():
    """Get the path to FFprobe executable"""
    # Try local FFprobe first
    local_ffprobe = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ffmpeg', 'ffprobe.exe')
    if os.path.exists(local_ffprobe):
        return local_ffprobe

    # Try alternative local paths
    alt_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ffmpeg', 'ffmpeg-master-latest-win64-gpl', 'bin', 'ffprobe.exe'),
    ]
    for path in alt_paths:
        if os.path.exists(path):
            return path

    # Fall back to system PATH
    return 'ffprobe'


def detect_fps(input_path):
    ffprobe_cmd = get_ffprobe_path()
    output = os.popen(f'"{ffprobe_cmd}" -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "{input_path}"').read()
    if "/" in output:
        try:
            return int(output.split("/")[0]) // int(output.split("/")[1].strip()), output.strip()
        except:
            pass
    return 30, 30


def run_ffmpeg(args):
    ffmpeg_cmd = get_ffmpeg_path()
    run_command(f'"{ffmpeg_cmd}" -hide_banner -hwaccel auto -loglevel {roop.globals.log_level} {args}')


def set_fps(input_path, output_path, fps):
    run_ffmpeg(f'-i "{input_path}" -filter:v fps=fps={fps} "{output_path}"')


def create_video(video_name, fps, output_dir):
    run_ffmpeg(f'-framerate "{fps}" -i "{output_dir}{os.sep}%04d.png" -c:v libx264 -crf 7 -pix_fmt yuv420p -y "{output_dir}{os.sep}output.mp4"')


def extract_frames(input_path, output_dir):
    run_ffmpeg(f'-i "{input_path}" "{output_dir}{os.sep}%04d.png"')


def add_audio(output_dir, video, keep_frames, output_file, index):
    video_name = os.path.splitext(video)[0]
    save_to = output_file + f"_{index+1}" + ".mp4" if output_file else output_dir + f"/swapped-" + video_name + f"_{index+1}" + ".mp4"
    run_ffmpeg(f'-i "{output_dir}{os.sep}output.mp4" -i "{output_dir}{os.sep}{video}" -c:v copy -map 0:v:0 -map 1:a:0 -y "{save_to}"')
    if not os.path.isfile(save_to):
        shutil.move(output_dir + "/output.mp4", save_to)
    if not keep_frames:
        shutil.rmtree(output_dir)


def is_img(path):
    return path.lower().endswith(("png", "jpg", "jpeg", "bmp"))
