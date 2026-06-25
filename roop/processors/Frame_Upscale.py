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


    @staticmethod
    def _tile_blend_mask(h: int, w: int, feather: int) -> np.ndarray:
        f = max(6, min(int(feather), h // 2, w // 2))
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, f, dtype=np.float32))
        mask_y = np.ones(h, dtype=np.float32)
        mask_x = np.ones(w, dtype=np.float32)
        mask_y[:f] *= ramp
        mask_y[-f:] *= ramp[::-1]
        mask_x[:f] *= ramp
        mask_x[-f:] *= ramp[::-1]
        return mask_y[:, None] * mask_x[None, :]

    def merge_tile_frames(self, tile_frames, temp_width : int, temp_height : int, pad_width : int, pad_height : int, size) -> Frame:
        tile_width = tile_frames[0].shape[1] - 2 * size[2]
        tiles_per_row = max(1, min(pad_width // max(tile_width, 1), len(tile_frames)))
        first_core = tile_frames[0][size[2]:-size[2], size[2]:-size[2]]
        th, tw = first_core.shape[:2]
        feather = max(32, min(th, tw) // 3, size[2] * 14)

        acc = np.zeros((pad_height, pad_width, 3), dtype=np.float64)
        weight = np.zeros((pad_height, pad_width), dtype=np.float64)

        for index, tile_frame in enumerate(tile_frames):
            core = tile_frame[size[2]:-size[2], size[2]:-size[2]]
            th, tw = core.shape[:2]
            row_index = index // tiles_per_row
            col_index = index % tiles_per_row
            top = row_index * th
            left = col_index * tw
            bottom = top + th
            right = left + tw
            mask = self._tile_blend_mask(th, tw, feather)
            acc[top:bottom, left:right, :] += core.astype(np.float64) * mask[..., None]
            weight[top:bottom, left:right] += mask

        weight = np.maximum(weight, 1e-6)
        merge_frame = (acc / weight[..., None]).clip(0, 255).astype(np.uint8)
        merge_frame = merge_frame[size[1]: size[1] + temp_height, size[1]: size[1] + temp_width, :]
        return merge_frame


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
        return (128, 8, 4)

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

