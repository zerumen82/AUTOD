import roop.globals
import cv2
import numpy as np
import onnx
import onnxruntime

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path


class FaceSwapInsightFace():
    plugin_options: dict = None
    model_swap_insightface = None

    processorname = 'faceswap'
    type = 'swap'

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_swap_insightface is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            graph = onnx.load(model_path).graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            self.input_mean = 0.0
            self.input_std = 255.0
            #cuda_options = {"arena_extend_strategy": "kSameAsRequested", 'cudnn_conv_algo_search': 'DEFAULT'}            
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False
            self.model_swap_insightface = onnxruntime.InferenceSession(model_path, sess_options,
                                                                       providers=roop.globals.execution_providers)

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        blob = cv2.dnn.blobFromImage(temp_frame, 1.0 / self.input_std, (128, 128),
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        io_binding = self.model_swap_insightface.io_binding()
        io_binding.bind_cpu_input("target", blob)
        io_binding.bind_cpu_input("source", latent)
        io_binding.bind_output("output", self.devicename)
        self.model_swap_insightface.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()[0]
        img_fake = ort_outs.transpose((0, 2, 3, 1))[0]
        return np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

        img_fake, M = self.model_swap_insightface.get(temp_frame, target_face, source_face, paste_back=False)

    #    target_face.matrix = M
    #    return img_fake 

    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None
