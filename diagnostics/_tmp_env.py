"""Report the interpreter and the versions of the modules the probes need."""
import sys

print("exe:", sys.executable)
for name in ("insightface", "onnxruntime", "cv2", "numpy", "PIL", "requests"):
    try:
        module = __import__(name)
        print(name, getattr(module, "__version__", "ok"))
    except BaseException as exc:  # noqa: BLE001 - diagnostic report
        print(name, "MISSING", exc)
