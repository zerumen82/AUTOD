"""Contrato v5.1222: la ruta de cámara vuelve a ~1 MP como la referencia.

Log real 2026-10-17 («cambia el pov», 16:50): producción alimentaba Qwen
con 768x512 directo y la LoRA de ángulos RECONSTRUÍA la escena sin moverla
(RUN2 00:11, RUN3 02:38 y RUN4 16:53: correlación 0.83 contra 0.35 de un
movimiento real en la oficina del 28/09). Misma entrada, misma pose y
misma semilla con FluxKontextImageScale (~1 MP): salida REENCUADRADA,
correlación -0.08, 197 s en la 8 GB (hoja:
diagnostics/out/pov_experiments_20261017.png). La workflow de referencia
del autor usa ese nodo; producción no lo tenía.

Dos grupos: (1) la workflow encola el escalado y VAEEncode/ image1 salen
de él; (2) el tope de subida solo acota fotos gigantes (la receta vieja
reescala TODO a 768 px, incluidas las que ya caben).
"""
import io
import time
from unittest.mock import patch

from PIL import Image

from roop.img_editor import qwen_camera_comfy_client as client_mod

# El tope nuevo aún no existe en rojo; el contrato lo exige en verde.
QwenCameraComfyClient = client_mod.QwenCameraComfyClient
UPLOAD_SIDE = getattr(client_mod, "UPLOAD_SIDE", None)


POSE = "<sks> front-right quarter view elevated shot medium shot"


def _png(size=(32, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size, "gray").save(buf, format="PNG")
    return buf.getvalue()


class _Resp:
    def __init__(self, payload=None, content=b""):
        self._payload = payload or {}
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def group_workflow():
    wf = QwenCameraComfyClient._workflow("in.png", POSE, 7)
    # El escalado a ~1 MP es el nodo 9 y todo sale de él.
    assert "9" in wf, sorted(wf)
    assert wf["9"]["class_type"] == "FluxKontextImageScale", wf["9"]
    assert wf["9"]["inputs"]["image"] == ["1", 0], wf["9"]
    assert wf["10"]["inputs"]["pixels"] == ["9", 0], wf["10"]
    assert wf["11"]["inputs"]["image1"] == ["9", 0], wf["11"]
    assert wf["12"]["inputs"]["image1"] == ["9", 0], wf["12"]
    # La receta que sí funcionó (oficina 28/09) queda intacta.
    assert wf["15"]["inputs"]["steps"] == 4, wf["15"]
    assert wf["15"]["inputs"]["denoise"] == 1.0, wf["15"]
    assert wf["15"]["inputs"]["seed"] == 7, wf["15"]
    assert wf["3"]["inputs"]["strength_model"] == 0.9, wf["3"]
    assert wf["11"]["inputs"]["prompt"] == POSE, wf["11"]
    print("OK: VAEEncode e image1 salen del escalado ~1 MP (FluxKontextImageScale)")


def _run_generate_once(image):
    """_generate_once real con la red y el watch simulados; captura subida."""
    captured = {}

    def fake_post(url, **kw):
        if str(url).endswith("/upload/image"):
            buf = kw["files"]["image"][1]
            buf.seek(0)
            with Image.open(buf) as uploaded:
                captured["upload"] = uploaded.size
            return _Resp({"name": "autoauto_camera.png"})
        if str(url).endswith("/prompt"):
            captured["workflow"] = kw["json"]["prompt"]
            return _Resp({"prompt_id": "probe-1"})
        raise AssertionError(f"POST inesperado: {url}")

    def fake_get(url, **kw):
        return _Resp(content=_png())

    with (patch("roop.img_editor.qwen_camera_comfy_client.requests.post",
                fake_post),
          patch("roop.img_editor.qwen_camera_comfy_client.requests.get",
                fake_get),
          patch("roop.img_editor.qwen_camera_comfy_client.wait_for_comfy_image",
                return_value=({"filename": "f.png", "subfolder": "",
                               "type": "output"}, "OK (stub)"))):
        result, message = QwenCameraComfyClient()._generate_once(
            "http://127.0.0.1:1", image, POSE, 7, None, None, time.time())
    assert result is not None, message
    return captured


def group_upload_cap():
    # Una foto que YA cabe no se reescala: el lienzo lo decide node9.
    captured = _run_generate_once(Image.new("RGB", (1920, 1280), "white"))
    assert captured["upload"] == (1920, 1280), captured.get("upload")
    wf = captured["workflow"]
    assert wf["9"]["class_type"] == "FluxKontextImageScale", wf.get("9")
    assert wf["10"]["inputs"]["pixels"] == ["9", 0], wf.get("10")
    # Una foto gigante sí se acota (subida/decode), con lado múltiplo de 16.
    big = _run_generate_once(Image.new("RGB", (4096, 3072), "white"))
    w, h = big["upload"]
    assert max(w, h) == UPLOAD_SIDE, big["upload"]
    assert w % 16 == 0 and h % 16 == 0, big["upload"]
    print(f"OK: subida intacta cuando cabe; tope de {UPLOAD_SIDE} px solo en gigantes")


def main():
    failures = []
    for group in (group_workflow, group_upload_cap):
        try:
            group()
        except AssertionError as error:
            failures.append(f"{group.__name__}: {error}")
            print(f"ROJO: {group.__name__}: {error}")
    if failures:
        raise SystemExit(f"{len(failures)} grupo(s) en rojo")
    print("TODO VERDE")


if __name__ == "__main__":
    main()
