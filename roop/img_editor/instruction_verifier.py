#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grounded, model-based verification of an image-edit instruction.

This deliberately has no catalogue of edit types.  The same visual-language
model receives the complete request and the rendered candidate, and decides
whether *all* requested changes are visibly present.
"""

import base64
import io
import json
import os
import re
from typing import Any, Dict, Optional

from PIL import Image


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class InstructionVerifier:
    """A small local VLM gate used after an image-edit candidate is rendered."""

    def __init__(self) -> None:
        self._llm = None
        self._handler = None
        self._unavailable_reason: Optional[str] = None

    def _load(self) -> bool:
        if self._llm is not None:
            return True
        if self._unavailable_reason is not None:
            return False

        root = _project_root()
        text_model = os.path.join(root, "models", "moondream", "moondream2-text-model-f16.gguf")
        vision_projector = os.path.join(root, "models", "moondream", "moondream2-mmproj-f16.gguf")
        if not (os.path.isfile(text_model) and os.path.isfile(vision_projector)):
            self._unavailable_reason = "modelos Moondream2 no encontrados"
            return False

        try:
            # Import lazily: a verifier must never prevent a normal edit when a
            # local llama build is unavailable.
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import MoondreamChatHandler

            # The projector has no device selector: the ggml backend decides.
            # On the CPU backend it aborts the process natively, so the vision
            # path is refused before any model is loaded.
            from roop.img_editor.vision_guard import check_vision_preflight

            preflight = check_vision_preflight()
            print(f"[InstructionVerifier] {preflight.as_message()}", flush=True)
            if not preflight.ok:
                self._unavailable_reason = preflight.as_message()
                return False

            self._handler = MoondreamChatHandler(clip_model_path=vision_projector, verbose=False)
            self._llm = Llama(
                model_path=text_model,
                chat_handler=self._handler,
                n_gpu_layers=0,
                n_ctx=1024,
                verbose=False,
            )
            print("[InstructionVerifier] Moondream2 listo (verificación visual local)", flush=True)
            return True
        except Exception as exc:
            self._unavailable_reason = str(exc)
            print(f"[InstructionVerifier] No disponible: {exc}", flush=True)
            self.unload()
            return False

    @staticmethod
    def _image_data_url(image: Image.Image) -> str:
        # The check needs semantics, not source resolution; this bound keeps the
        # VLM prompt cheap without changing the generated image.
        preview = image.convert("RGB").copy()
        preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        preview.save(buf, format="JPEG", quality=90)
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    @staticmethod
    def _parse_answer(text: str) -> Dict[str, Any]:
        value = (text or "").strip()
        # Moondream is a caption/VQA model, not an instruction-tuned JSON API.
        # A one-token protocol is therefore accepted in addition to JSON.
        # These are protocol tokens, not knowledge of any image-edit request.
        first_token = re.split(r"\s+|[.:,;!?]", value.upper(), maxsplit=1)[0]
        if first_token in ("PASS", "YES"):
            return {"available": True, "compliant": True, "unmet": "", "raw": value}
        if first_token in ("FAIL", "NO"):
            return {"available": True, "compliant": False, "unmet": "", "raw": value}
        match = re.search(r"\{.*?\}", value, flags=re.DOTALL)
        if not match:
            return {"available": True, "compliant": None, "raw": value}
        try:
            data = json.loads(match.group(0))
        except Exception:
            return {"available": True, "compliant": None, "raw": value}
        if not isinstance(data, dict) or not isinstance(data.get("pass"), bool):
            return {"available": True, "compliant": None, "raw": value}
        unmet = data.get("unmet", "")
        if isinstance(unmet, list):
            unmet = "; ".join(str(x) for x in unmet)
        return {
            "available": True,
            "compliant": bool(data["pass"]),
            "unmet": str(unmet or "").strip(),
            "raw": value,
        }

    def verify(self, image: Image.Image, request: str) -> Dict[str, Any]:
        """Return a tri-state verdict: true, false, or None when inconclusive."""
        if not request or not request.strip() or not self._load():
            return {"available": False, "compliant": None, "reason": self._unavailable_reason}
        try:
            question = (
                "Inspect this edited image and assess the complete edit request below. "
                "Do not assume a requested change happened unless it is visibly clear. "
                "All independent requested changes must be satisfied for a pass. "
                "Reply with exactly one word: PASS if every requested change is visibly satisfied; otherwise FAIL.\n\n"
                f"Edit request: {request.strip()}"
            )
            answer = self._llm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._image_data_url(image)}},
                        {"type": "text", "text": question},
                    ],
                }],
                temperature=0.0,
                max_tokens=100,
                stop=["\n\nQuestion:"],
            )
            content = answer["choices"][0]["message"]["content"]
            verdict = self._parse_answer(content)
            if verdict.get("compliant") is True:
                print(
                    "[InstructionVerifier] Señal visual positiva del VLM local "
                    "(no garantiza todos los detalles; inspeccionar el resultado)",
                    flush=True,
                )
            elif verdict.get("compliant") is False:
                print(f"[InstructionVerifier] FAIL: {verdict.get('unmet') or 'requisitos no visibles'}", flush=True)
            else:
                print(f"[InstructionVerifier] Veredicto no concluyente: {str(content)[:240]!r}", flush=True)
            return verdict
        except Exception as exc:
            print(f"[InstructionVerifier] Verificación omitida: {exc}", flush=True)
            return {"available": False, "compliant": None, "reason": str(exc)}
        finally:
            # Keep the verifier from competing with the next ComfyUI invocation.
            self.unload()

    def unload(self) -> None:
        handler, llm = self._handler, self._llm
        self._handler = None
        self._llm = None
        try:
            if llm is not None:
                del llm
            if handler is not None:
                close = getattr(handler, "close", None)
                if callable(close):
                    close()
        finally:
            try:
                import gc
                gc.collect()
            except Exception:
                pass


_verifier: Optional[InstructionVerifier] = None


def get_instruction_verifier() -> InstructionVerifier:
    global _verifier
    if _verifier is None:
        _verifier = InstructionVerifier()
    return _verifier
