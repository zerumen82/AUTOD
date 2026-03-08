from typing import Any, List, Callable
import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.types import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path

class Enhance_RestoreFormerPPlus():
    plugin_options:dict = None
    model_restoreformerpplus = None
    devicename = None
    name = None

    processorname = 'restoreformer++'
    type = 'enhance'
    

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_restoreformerpplus is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            model_path = resolve_relative_path('../models/restoreformer_plus_plus.onnx')
            self.model_restoreformerpplus = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_restoreformerpplus.get_inputs()
            model_outputs = self.model_restoreformerpplus.get_outputs()
            self.io_binding = self.model_restoreformerpplus.io_binding()
            self.io_binding.bind_output(model_outputs[0].name, self.devicename)

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        # preprocess - guardar tamaño original
        original_height, original_width = temp_frame.shape[:2]
        input_size = temp_frame.shape[1]
        
        # Resize a 512x512 para el modelo
        temp_frame = cv2.resize(temp_frame, (512, 512), cv2.INTER_CUBIC)
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)
        
        self.io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame) # .astype(np.float32)
        self.model_restoreformerpplus.run_with_iobinding(self.io_binding)
        ort_outs = self.io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs 
        
        # Postprocess
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        # El resultado es 512x512, calcular scale_factor correcto
        # scale_factor indica cuánto se amplió la imagen (512 / input_size)
        scale_factor = 512.0 / input_size if input_size > 0 else 1.0
        
        # Retornar resultado 512x512 y el scale_factor
        # El llamador hará el resize al tamaño original
        return result.astype(np.uint8), scale_factor


    def Release(self):
        del self.model_restoreformerpplus
        self.model_restoreformerpplus = None
        del self.io_binding
        self.io_binding = None

