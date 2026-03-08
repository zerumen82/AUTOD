import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional

class AnimatePhoto:
    def __init__(self, comfy_host="http://127.0.0.1", comfy_port=8188):
        self.comfy_host = comfy_host
        self.comfy_port = comfy_port
        self.base_url = f"{comfy_host}:{comfy_port}/api/v1"
        self.session = requests.Session()
        self.queue_id = None
        
    def check_comfyui_status(self) -> bool:
        """Verifica si ComfyUI está corriendo"""
        try:
            response = self.session.get(f"{self.base_url}/queue", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def upload_image(self, image_path: str) -> str:
        """Sube una imagen a ComfyUI"""
        url = f"{self.base_url}/upload"
        files = {'file': open(image_path, 'rb')}
        response = self.session.post(url, files=files)
        data = response.json()
        return data['path'] if 'path' in data else None
    
    def create_workflow(self, model: str, image_path: str, prompt: str, 
                      output_path: str, frames: int = 24, fps: int = 24) -> Dict:
        """Crea un workflow JSON para el modelo especificado"""
        
        workflow = {
            "description": f"Workflow {model} - {prompt[:50]}...",
            "nodes": []
        }
        
        if model == "svd_turbo":
            workflow["nodes"] = self._create_svd_workflow(image_path, prompt, output_path, frames, fps)
        elif model == "wan2.2":
            workflow["nodes"] = self._create_wan_workflow(image_path, prompt, output_path, frames, fps)
        elif model == "zeroscope":
            workflow["nodes"] = self._create_zeroscope_workflow(image_path, prompt, output_path, frames, fps)
        elif model == "ltxv":
            workflow["nodes"] = self._create_ltxv_workflow(image_path, prompt, output_path, frames, fps)
        else:
            raise ValueError(f"Modelo {model} no soportado")
        
        return workflow
    
    def _create_svd_workflow(self, image_path: str, prompt: str, output_path: str,
                           frames: int, fps: int) -> List[Dict]:
        """Crea workflow para SVD Turbo"""
        return [
            {
                "id": "load_image",
                "type": "load_image",
                "params": {"image_path": image_path},
                "inputs": {}, "outputs": {"image": "image"}}
            ,
            {
                "id": "vae_loader",
                "type": "vae_loader",
                "params": {"vae_model_path": "models/vae/svd_xt_image_decoder.safetensors"},
                "inputs": {}, "outputs": {"vae": "vae"}}
            ,
            {
                "id": "clip_vision",
                "type": "clip_vision_loader",
                "params": {"clip_vision_model_path": "models/clip_vision/open_clip_pytorch_model.bin"},
                "inputs": {}, "outputs": {"clip_vision": "clip_vision"}}
            ,
            {
                "id": "unet_loader",
                "type": "unet_loader",
                "params": {"model_path": "models/diffusion_models/StableDiffusionTurbo/svd_xt.safetensors"},
                "inputs": {}, "outputs": {"unet": "unet"}}
            ,
            {
                "id": "conditioning",
                "type": "svd_img2vid_conditioning",
                "params": {
                    "prompt": prompt,
                    "negative_prompt": "low quality, bad anatomy, worst quality, lowres, text, error, cropped",
                    "resolution": "720x480",
                    "frames": frames,
                    "fps": fps,
                    "motion_bucket_id": 127,
                    "fps_id": 5
                },
                "inputs": {
                    "image": "image",
                    "clip_vision": "clip_vision"
                },
                "outputs": {"latent_image": "latent_image", "noise": "noise"}}
            ,
            {
                "id": "sampler",
                "type": "sampler",
                "params": {
                    "steps": 20,
                    "cfg_scale": 3.0,
                    "scheduler": "sgm_uniform",
                    "denoise": 1.0,
                    "sampler_name": "euler_ancestral"
                },
                "inputs": {
                    "model": "unet",
                    "noise": "noise",
                    "latent_image": "latent_image"
                },
                "outputs": {"latent": "latent"}}
            ,
            {
                "id": "vae_decode",
                "type": "vae_decode",
                "params": {},
                "inputs": {
                    "vae": "vae",
                    "latent": "latent"
                },
                "outputs": {"image": "image"}}
            ,
            {
                "id": "create_video",
                "type": "create_video",
                "params": {
                    "output_path": output_path,
                    "fps": fps
                },
                "inputs": {
                    "image": "image"
                },
                "outputs": {}}
        ]
    
    def _create_wan_workflow(self, image_path: str, prompt: str, output_path: str,
                           frames: int, fps: int) -> List[Dict]:
        """Crea workflow para Wan2.2-Animate"""
        return [
            {
                "id": "wan_loader",
                "type": "wan_video_model_loader",
                "params": {"model_path": "models/diffusion_models/Wan2.2-Animate-14B-Q2_K.gguf"},
                "inputs": {}, "outputs": {"model": "model"}}
            ,
            {
                "id": "text_encode",
                "type": "wan_video_text_encode",
                "params": {"prompt": prompt},
                "inputs": {}, "outputs": {"text_encoding": "text_encoding"}}
            ,
            {
                "id": "image_loader",
                "type": "load_image",
                "params": {"image_path": image_path},
                "inputs": {}, "outputs": {"image": "image"}}
            ,
            {
                "id": "encode",
                "type": "wan_video_encode",
                "params": {
                    "resolution": "720x480",
                    "frames": frames,
                    "fps": fps,
                    "guidance": 6.0
                },
                "inputs": {
                    "image": "image",
                    "text_encoding": "text_encoding"
                },
                "outputs": {"video": "video"}}
            ,
            {
                "id": "sampling",
                "type": "wan_image_to_video",
                "params": {
                    "steps": 30,
                    "guidance_scale": 6.0,
                    "scheduler": "ddim"
                },
                "inputs": {
                    "model": "model",
                    "video": "video"
                },
                "outputs": {"output": "output"}}
            ,
            {
                "id": "save_video",
                "type": "save_video",
                "params": {"output_path": output_path},
                "inputs": {
                    "video": "output"
                },
                "outputs": {}}
        ]
    
    def _create_zeroscope_workflow(self, image_path: str, prompt: str, output_path: str,
                                 frames: int, fps: int) -> List[Dict]:
        """Crea workflow para Zeroscope XL"""
        return [
            {
                "id": "zeroscope_loader",
                "type": "zeroscope_video_model_loader",
                "params": {
                    "model_path": "models/diffusion_models/zeroscope_v2_XL/checkpoints/zeroscope_v2_XL.safetensors",
                    "text_encoder_path": "models/diffusion_models/zeroscope_v2_XL/text_encoder/pytorch_model.bin",
                    "tokenizer_path": "models/diffusion_models/zeroscope_v2_XL/tokenizer/vocab.json"
                },
                "inputs": {}, "outputs": {"model": "model", "text_encoder": "text_encoder", "tokenizer": "tokenizer"}}
            ,
            {
                "id": "text_encode",
                "type": "text_encode",
                "params": {"prompt": prompt},
                "inputs": {
                    "text_encoder": "text_encoder",
                    "tokenizer": "tokenizer"
                },
                "outputs": {"text_encoding": "text_encoding"}}
            ,
            {
                "id": "image_loader",
                "type": "load_image",
                "params": {"image_path": image_path},
                "inputs": {}, "outputs": {"image": "image"}}
            ,
            {
                "id": "sampling",
                "type": "sampler",
                "params": {
                    "steps": 25,
                    "cfg_scale": 4.0,
                    "scheduler": "sgm_uniform",
                    "denoise": 1.0,
                    "sampler_name": "euler_ancestral"
                },
                "inputs": {
                    "model": "model",
                    "prompt": "text_encoding",
                    "image": "image"
                },
                "outputs": {"output": "output"}}
            ,
            {
                "id": "save_video",
                "type": "save_video",
                "params": {
                    "output_path": output_path,
                    "fps": fps,
                    "frames": frames,
                    "resolution": "576x320"
                },
                "inputs": {
                    "video": "output"
                },
                "outputs": {}}
        ]
    
    def _create_ltxv_workflow(self, image_path: str, prompt: str, output_path: str,
                            frames: int, fps: int) -> List[Dict]:
        """Crea workflow para LTXV"""
        return [
            {
                "id": "ltxv_loader",
                "type": "ltxv_video_model_loader",
                "params": {
                    "model_path": "models/diffusion_models/ltxv/checkpoints/ltxv.safetensors",
                    "text_encoder_path": "models/diffusion_models/ltxv/text_encoder/pytorch_model.bin",
                    "tokenizer_path": "models/diffusion_models/ltxv/tokenizer/vocab.json"
                },
                "inputs": {}, "outputs": {"model": "model", "text_encoder": "text_encoder", "tokenizer": "tokenizer"}}
            ,
            {
                "id": "text_encode",
                "type": "text_encode",
                "params": {"prompt": prompt},
                "inputs": {
                    "text_encoder": "text_encoder",
                    "tokenizer": "tokenizer"
                },
                "outputs": {"text_encoding": "text_encoding"}}
            ,
            {
                "id": "image_loader",
                "type": "load_image",
                "params": {"image_path": image_path},
                "inputs": {}, "outputs": {"image": "image"}}
            ,
            {
                "id": "sampling",
                "type": "sampler",
                "params": {
                    "steps": 30,
                    "cfg_scale": 5.0,
                    "scheduler": "ddim",
                    "denoise": 1.0,
                    "sampler_name": "euler_ancestral"
                },
                "inputs": {
                    "model": "model",
                    "prompt": "text_encoding",
                    "image": "image"
                },
                "outputs": {"output": "output"}}
            ,
            {
                "id": "save_video",
                "type": "save_video",
                "params": {
                    "output_path": output_path,
                    "fps": fps,
                    "frames": frames,
                    "resolution": "720x480"
                },
                "inputs": {
                    "video": "output"
                },
                "outputs": {}}
        ]
    
    def queue_workflow(self, workflow: Dict) -> str:
        """Encola el workflow en ComfyUI"""
        url = f"{self.base_url}/queue"
        response = self.session.post(url, json=workflow)
        data = response.json()
        self.queue_id = data['id']
        return self.queue_id
    
    def get_queue_status(self) -> Dict:
        """Obtiene el estado de la cola"""
        if not self.queue_id:
            return {"error": "No hay queue_id"}
        
        url = f"{self.base_url}/queue/{self.queue_id}"
        response = self.session.get(url)
        return response.json()
    
    def get_all_queues(self) -> List[Dict]:
        """Obtiene todas las colas"""
        url = f"{self.base_url}/queue"
        response = self.session.get(url)
        return response.json()
    
    def animate_image(self, model: str, image_path: str, prompt: str,
                    output_path: str, frames: int = 24, fps: int = 24,
                    timeout: int = 300) -> bool:
        """Función principal para animar una imagen"""
        
        print(f"[INFO] Iniciando animación con {model.upper()}...")
        print(f"[INFO] Prompt: {prompt}")
        print(f"[INFO] Imagen: {image_path}")
        print(f"[INFO] Salida: {output_path}")
        
        # Verificar ComfyUI
        if not self.check_comfyui_status():
            print("[ERROR] ComfyUI no está corriendo en el puerto 8188")
            return False
        
        # Subir imagen
        print("[INFO] Subiendo imagen...")
        uploaded_path = self.upload_image(image_path)
        if not uploaded_path:
            print("[ERROR] No se pudo subir la imagen")
            return False
        
        # Crear workflow
        print("[INFO] Creando workflow...")
        workflow = self.create_workflow(model, uploaded_path, prompt, output_path, frames, fps)
        
        # Encolar workflow
        print("[INFO] Encolando workflow...")
        queue_id = self.queue_workflow(workflow)
        
        # Monitorear progreso
        print("[INFO] Procesando... (esto puede tardar varios minutos)")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_queue_status()
            
            if status.get('status') == 'complete':
                print(f"[INFO] Animación completada exitosamente!")
                print(f"[INFO] Archivo guardado en: {output_path}")
                return True
            elif status.get('status') == 'error':
                print(f"[ERROR] Error en el procesamiento: {status.get('error', 'Desconocido')}")
                return False
            
            # Mostrar progreso
            progress = status.get('progress', 0)
            eta = status.get('eta', 'desconocido')
            print(f"[INFO] Progreso: {progress:.1f}% | ETA: {eta}", end='\r')
            
            time.sleep(5)
        
        print("[ERROR] Tiempo de espera agotado")
        return False

def main():
    # Crear instancia de AnimatePhoto
    animator = AnimatePhoto()
    
    # Ejemplo de uso
    model = "ltxv"  # Opciones: "svd_turbo", "wan2.2", "zeroscope", "ltxv"
    image_path = "testdata/test1.jpg"
    prompt = "Un paisaje urbano futurista con coches voladores y rascacielos iluminados"
    output_path = "animacion_output.mp4"
    
    # Ejecutar animación
    success = animator.animate_image(
        model=model,
        image_path=image_path,
        prompt=prompt,
        output_path=output_path,
        frames=24,
        fps=24,
        timeout=600
    )
    
    if success:
        print("\n[ÉXITO] Animación completada!")
    else:
        print("\n[FALLO] La animación no se completó")

if __name__ == "__main__":
    main()