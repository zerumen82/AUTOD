import cv2
import numpy as np
import onnxruntime
import roop.globals
from roop.types import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path

class Enhance_GFPGAN():
    plugin_options: dict = None
    model_gfpgan = None
    name = None
    devicename = None
    processorname = 'gfpgan'
    type = 'enhance'

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_gfpgan is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.onnx')
            
            # Determine providers based on plugin_options
            providers = []
            devnm = plugin_options.get("devicename", "cpu").lower()
            if 'cuda' in devnm:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
                
            try:
                self.model_gfpgan = onnxruntime.InferenceSession(model_path, None, providers=providers)
                self.devicename = devnm
                print(f"[INFO] Enhance_GFPGAN initialized (providers={providers})")
            except Exception as e:
                print(f"[ERROR] Failed to initialize GFPGAN with {providers}: {e}")
                if 'cuda' in devnm:
                    print("[INFO] Retrying GFPGAN with CPU...")
                    try:
                        self.model_gfpgan = onnxruntime.InferenceSession(model_path, None, providers=['CPUExecutionProvider'])
                        self.devicename = 'cpu'
                        print("[INFO] Enhance_GFPGAN initialized with CPU")
                    except Exception as e2:
                        print(f"[ERROR] Failed fallback to CPU for GFPGAN: {e2}")

        if self.model_gfpgan is not None:
            self.name = self.model_gfpgan.get_inputs()[0].name

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        # preprocess
        input_size = temp_frame.shape[1]

        # Resize the input image to 512x512
        temp_frame = cv2.resize(temp_frame, (512, 512), cv2.INTER_CUBIC)

        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)

        io_binding = self.model_gfpgan.io_binding()
        io_binding.bind_cpu_input("input", temp_frame)
        io_binding.bind_output("1288", self.devicename)
        self.model_gfpgan.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]

        # post-process
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        scale_factor = int(result.shape[1] / input_size)
        return result.astype(np.uint8), scale_factor

    def Release(self):
        self.model_gfpgan = None











