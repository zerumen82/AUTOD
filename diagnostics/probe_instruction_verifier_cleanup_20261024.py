"""Resource lifetime regression: projector buffers must close before next edit."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor.instruction_verifier import InstructionVerifier  # noqa: E402


def run(fail_text_close=False):
    released = []

    def close_text():
        released.append("text")
        if fail_text_close:
            raise RuntimeError("diagnostic text close failure")

    stack = ExitStack()
    stack.callback(released.append, "projector")
    verifier = InstructionVerifier()
    verifier._llm = SimpleNamespace(close=close_text)
    verifier._handler = SimpleNamespace(_exit_stack=stack)
    verifier.unload()
    assert released == ["text", "projector"], released
    assert verifier._llm is None and verifier._handler is None
    verifier.unload()
    assert released == ["text", "projector"], "cleanup was not idempotent"


if __name__ == "__main__":
    for failure in (False, True):
        run(failure)
        print(f"PASS text_close_failure={failure}", flush=True)
