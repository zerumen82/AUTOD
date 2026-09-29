import cv2 
import numpy as np
import onnxruntime
import roop.globals
import threading

from roop.utilities import resolve_relative_path
from roop.types import Frame

class Frame_Upscale():
    plugin_options:dict = None
    model_upscale = None
    devicename = None
    prev_type = None

    processorname = 'upscale'
    type = 'frame_enhancer'

    THREAD_LOCK_UPSCALE = threading.Lock()


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.prev_type is not None and self.prev_type != self.plugin_options["subtype"]:
            self.Release()
        self.prev_type = self.plugin_options["subtype"]
        if self.model_upscale is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            if self.prev_type == "esrganx4":
                model_path = resolve_relative_path('../models/Frame/real_esrgan_x4.onnx')
                self.scale = 4
            elif self.prev_type == "esrganx2":
                model_path = resolve_relative_path('../models/Frame/real_esrgan_x2.onnx')
                self.scale = 2
            elif self.prev_type == "lsdirx4":
                model_path = resolve_relative_path('../models/Frame/lsdir_x4.onnx')
                self.scale = 4

            self.model_upscale = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_upscale.get_inputs()
            model_outputs = self.model_upscale.get_outputs()
            self.io_binding = self.model_upscale.io_binding()
            self.io_binding.bind_output(model_outputs[0].name, self.devicename)

    def getProcessedResolution(self, width, height):
        return (width * self.scale, height * self.scale)

# borrowed from facefusion -> https://github.com/facefusion/facefusion
    def prepare_tile_frame(self, tile_frame : Frame) -> Frame:
        tile_frame = np.expand_dims(tile_frame[:, :, ::-1], axis = 0)
        tile_frame = tile_frame.transpose(0, 3, 1, 2)
        tile_frame = tile_frame.astype(np.float32) / 255
        return tile_frame


    def normalize_tile_frame(self, tile_frame : Frame) -> Frame:
        tile_frame = tile_frame.transpose(0, 2, 3, 1).squeeze(0) * 255
        tile_frame = tile_frame.clip(0, 255).astype(np.uint8)[:, :, ::-1]
        return tile_frame

    def create_tile_frames(self, input_frame : Frame, size):
        input_frame = np.pad(input_frame, ((size[1], size[1]), (size[1], size[1]), (0, 0)))
        tile_width = size[0] - 2 * size[2]
        pad_size_bottom = size[2] + tile_width - input_frame.shape[0] % tile_width
        pad_size_right = size[2] + tile_width - input_frame.shape[1] % tile_width
        pad_vision_frame = np.pad(input_frame, ((size[2], pad_size_bottom), (size[2], pad_size_right), (0, 0)))
        pad_height, pad_width = pad_vision_frame.shape[:2]
        row_range = range(size[2], pad_height - size[2], tile_width)
        col_range = range(size[2], pad_width - size[2], tile_width)
        tile_frames = []

        for row_frame in row_range:
            top = row_frame - size[2]
            bottom = row_frame + size[2] + tile_width
            for column_vision_frame in col_range:
                left = column_vision_frame - size[2]
                right = column_vision_frame + size[2] + tile_width
                tile_frames.append(pad_vision_frame[top:bottom, left:right, :])
        return tile_frames, pad_width, pad_height


    def merge_tile_frames(self, tile_frames, temp_width : int, temp_height : int, pad_width : int, pad_height : int, size) -> Frame:
        """
        Merge con solape REAL: cada tile incluye 2*size[2] px de contexto y los tiles
        vecinos COMPARTEN esa banda. Se coloca cada tile en su posición verdadera del
        lienzo y se funde solo sobre la banda de solape, con un ramp coseno con SUELO
        (>0 siempre). Antes los tiles se colocaban ABUTTING con un feather que llegaba
        a peso exacto 0 en el borde => acc/weight = 0/eps = 0 => REJILLA NEGRA con
        cruces en las intersecciones (bug v5.1132 y anteriores).
        """
        full_h, full_w = tile_frames[0].shape[:2]
        step_h = full_h - 2 * size[2]   # paso entre orígenes de tiles (núcleo sin pad), escalado
        step_w = full_w - 2 * size[2]
        cols = 1 + int(np.ceil(max(0, pad_width - full_w) / max(step_w, 1) - 1e-6))

        # Feather SOLO en la banda de solape real (2*size[2] escalado), con suelo 0.02:
        # píxel cubierto por un solo tile => normalización devuelve su propio valor;
        # píxel en solape => media suave de ambos (mismo contexto, costura invisible).
        feather = max(2, min(2 * size[2], step_h // 4, step_w // 4))
        ramp = 0.02 + 0.98 * (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, feather, dtype=np.float32)))

        acc = np.zeros((pad_height, pad_width, 3), dtype=np.float64)
        weight = np.zeros((pad_height, pad_width), dtype=np.float64)

        for index, tile_frame in enumerate(tile_frames):
            row_index = index // cols
            col_index = index % cols
            top = min(row_index * step_h, pad_height - full_h) if pad_height >= full_h else 0
            left = min(col_index * step_w, pad_width - full_w) if pad_width >= full_w else 0
            bottom = min(top + full_h, pad_height)
            right = min(left + full_w, pad_width)
            tile = tile_frame[: bottom - top, : right - left].astype(np.float64)
            th, tw = bottom - top, right - left

            my = np.ones(th, dtype=np.float64)
            mx = np.ones(tw, dtype=np.float64)
            f_h = min(feather, th // 2)
            f_w = min(feather, tw // 2)
            my[:f_h] *= ramp[:f_h]
            my[th - f_h:] *= ramp[:f_h][::-1]
            mx[:f_w] *= ramp[:f_w]
            mx[tw - f_w:] *= ramp[:f_w][::-1]
            m = my[:, None] * mx[None, :]

            acc[top:bottom, left:right, :] += tile * m[..., None]
            weight[top:bottom, left:right] += m

        weight = np.maximum(weight, 1e-6)
        merge_frame = (acc / weight[..., None]).clip(0, 255).astype(np.uint8)
        # Contenido de la imagen en el lienzo: pads exterior (size[1]) + interior (size[2]).
        y0 = size[1] + size[2]
        x0 = size[1] + size[2]
        return merge_frame[y0 : y0 + temp_height, x0 : x0 + temp_width, :]


    @staticmethod
    def _pick_tile_size(temp_height: int, temp_width: int, scale: int = 2) -> tuple:
        """
        Tiles según escala del modelo y VRAM.
        x4 (LSDIR/ESRGAN): tiles pequeños — 512px input → 2048px out cuelga en 8GB.
        x2: tiles más grandes = menos costuras.
        """
        max_dim = max(temp_height, temp_width)
        if scale >= 4:
            # ~120px núcleo → ~480px salida por tile (seguro en RTX 3060 Ti 8GB)
            return (128, 8, 6)
        if max_dim >= 1200:
            return (384, 14, 12)
        if max_dim >= 640:
            return (256, 12, 10)
        # x2 on small photos was split into 20-30 tiny tiles, spending more
        # time in ONNX calls and stitching than in useful detail recovery.
        # 256px is already used safely by the same model on larger photos.
        return (256, 12, 10)

    def Run(self, temp_frame: Frame) -> Frame:
        temp_height, temp_width = temp_frame.shape[:2]
        scale = int(getattr(self, "scale", 2) or 2)
        size = self._pick_tile_size(temp_height, temp_width, scale=scale)
        upscale_tile_frames, pad_width, pad_height = self.create_tile_frames(temp_frame, size)
        total = len(upscale_tile_frames)
        label = self.prev_type or "upscale"
        print(
            f"[Frame_Upscale] {label} {scale}x — {temp_width}×{temp_height}, "
            f"{total} tile(s), tile={size[0]}px",
            flush=True,
        )

        for index, tile_frame in enumerate(upscale_tile_frames):
            if index == 0 or (index + 1) % max(1, total // 8) == 0 or index + 1 == total:
                print(f"[Frame_Upscale] Progreso {index + 1}/{total}", flush=True)
            tile_frame = self.prepare_tile_frame(tile_frame)
            with self.THREAD_LOCK_UPSCALE:
                self.io_binding.bind_cpu_input(self.model_inputs[0].name, tile_frame)
                self.model_upscale.run_with_iobinding(self.io_binding)
                ort_outs = self.io_binding.copy_outputs_to_cpu()
                result = ort_outs[0]
            upscale_tile_frames[index] = self.normalize_tile_frame(result)
        final_frame = self.merge_tile_frames(upscale_tile_frames, temp_width * self.scale
                                                    , temp_height * self.scale
                                                    , pad_width * self.scale, pad_height * self.scale
                                                    , (size[0] * self.scale, size[1] * self.scale, size[2] * self.scale))
        return final_frame.astype(np.uint8)



    def Release(self):
        del self.model_upscale
        self.model_upscale = None
        del self.io_binding
        self.io_binding = None
