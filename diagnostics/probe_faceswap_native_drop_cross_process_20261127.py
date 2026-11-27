"""Exercise FaceSwap native-drop queues across a secondary and primary process.

Uses the running local UI, adds two known files, and clears both banks after
the probe. It does not run FaceSwap or Image Editor inference.
"""

from pathlib import Path
import subprocess
import sys

from gradio_client import Client

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\PROJECTS\uee\FACES\AATL\grok_image_1773140604875.jpg")
TARGET = Path(r"D:\PROJECTS\uee\FACES\AATL\paloma.jpg")


def main():
    assert SOURCE.is_file() and TARGET.is_file(), (SOURCE, TARGET)
    client = Client("http://127.0.0.1:7861", verbose=False)
    producer = (
        "import roop.globals as g; "
        "g.native_drop_push('faceswap_src', [" + repr(str(SOURCE)) + "]); "
        "g.native_drop_push('faceswap_dest', [" + repr(str(TARGET)) + "])"
    )
    try:
        subprocess.run(
            [sys.executable, "-c", producer], cwd=ROOT, check=True,
        )
        source_result = client.predict(api_name="/poll_native_src_drop")
        target_result = client.predict(api_name="/poll_native_dest_drop")
        assert source_result[0], "no source faces returned"
        assert SOURCE.name in source_result[-1], "source file not registered"
        assert target_result[1], "no destination preview returned"
        assert TARGET.name in target_result[-1], "destination file not registered"
        print(
            f"PASS: {len(source_result[0])} source face(s), "
            "destination preview, both files registered cross-process"
        )
    finally:
        client.predict(api_name="/clear_source")
        client.predict(api_name="/clear_target")


if __name__ == "__main__":
    main()
