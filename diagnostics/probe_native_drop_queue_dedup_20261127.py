"""Verify native drag queue delivery within one process and across polls."""

from pathlib import Path
import tempfile

import roop.globals as state


def main():
    original_dir = state._native_drop_dir
    with tempfile.TemporaryDirectory(prefix="autoauto_drop_dedup_") as temporary:
        state._native_drop_dir = lambda: temporary
        try:
            source = str(Path(temporary) / "source.jpg")
            target = str(Path(temporary) / "target.jpg")
            state.native_drop_push("faceswap_src", [source, source])
            state.native_drop_push("faceswap_dest", [target])
            assert state.native_drop_pop("faceswap_src") == [source]
            assert state.native_drop_pop("faceswap_dest") == [target]
            assert state.native_drop_pop("faceswap_src") == []
            assert state.native_drop_pop("faceswap_dest") == []
            assert list(Path(temporary).glob("*.json")) == []
        finally:
            state._native_drop_dir = original_dir
    print("PASS: same-process JSON+memory delivery is unique and consumed once")


if __name__ == "__main__":
    main()
