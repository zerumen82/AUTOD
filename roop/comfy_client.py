#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Client - Cliente para interactuar con ComfyUI via API REST
"""

import os
import time
import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from io import BytesIO

# Puerto por defecto
DEFAULT_COMFY_PORT = '8188'

# Detectar puerto automaticamente
def get_comfyui_port():
    """Detecta el puerto de ComfyUI automaticamente"""
    possible_ports = ['8188', '8189', '8190', '8888', '8000']
    
    for port in possible_ports:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/system_stats", timeout=1)
            if response.status_code == 200:
                return port
        except:
            continue
    
    # Si no se detecta, usar variable de entorno o defecto
    return os.environ.get('COMFYUI_PORT', DEFAULT_COMFY_PORT)


def get_comfyui_url():
    """Obtiene la URL de ComfyUI"""
    port = get_comfyui_port()
    return f"http://127.0.0.1:{port}"


# URL dinamica
def _get_comfy_url():
    """Obtiene la URL de ComfyUI (para usar en funciones)"""
    return get_comfyui_url()

TIMEOUT = 300  # 5 minutos para generacion


class ComfyClient:
    """Cliente para ComfyUI"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or get_comfyui_url()
        self.client_id = f"comfy_client_{int(time.time())}"
    
    def get_history(self, prompt_id: str) -> Dict:
        """Obtiene el historial de un prompt"""
        try:
            response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def get_images(self, prompt_id: str, node_id: str = "3") -> list:
        """Obtiene imagens de un prompt"""
        images = []
        
        print(f"[ComfyClient] get_images: {prompt_id[:8]}...")
        
        max_attempts = 1200  # Maximo 600 segundos esperando (10 minutos) - primera carga es lenta
        attempts = 0
        last_status = ""
        found_images = False
        nan_detected = False
        
        while attempts < max_attempts:
            attempts += 1
            
            try:
                history = self.get_history(prompt_id)
            except Exception as e:
                print(f"[ComfyClient] Error getting history: {e}")
                time.sleep(0.5)
                continue
            
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                status = history[prompt_id].get("status", "")
                
                # Verificar si hay errores NaN en los outputs
                for nid, node_output in outputs.items():
                    if "errors" in node_output:
                        for error in node_output["errors"]:
                            error_msg = str(error.get("message", ""))
                            if "nan" in error_msg.lower() or "NaN" in error_msg:
                                print(f"[ComfyClient] ERROR NaN detectado en nodo {nid}: {error_msg}")
                                nan_detected = True
                    # Verificar si hay datos inválidos en las imágenes
                    if "images" in node_output:
                        for img in node_output["images"]:
                            if img.get("filename", "").lower().endswith("nan"):
                                print(f"[ComfyClient] ERROR: Archivo con nombre NaN detectado")
                                nan_detected = True
                
                if nan_detected:
                    raise Exception("Error: El modelo generó valores NaN. Esto suele ocurrir por incompatibilidad con el VAE o el modelo. Prueba con otro VAE o modelo.")
                
                # Mostrar progreso cada 10 intentos
                if attempts % 20 == 0:
                    print(f"[ComfyClient] Esperando... intento {attempts}, status={status}")
                
                if outputs:
                    print(f"[ComfyClient] Outputs disponibles: {list(outputs.keys())}")
                    
                    # Buscar imágenes en todos los nodos
                    for nid, node_output in outputs.items():
                        if "images" in node_output:
                            print(f"[ComfyClient] ¡IMAGEN ENCONTRADA en nodo {nid}!")
                            for image in node_output["images"]:
                                print(f"[ComfyClient] Descargando: {image['filename']}")
                                image_data = requests.get(
                                    f"{self.base_url}/view",
                                    params={
                                        "filename": image["filename"],
                                        "subfolder": image["subfolder"],
                                        "type": image["type"]
                                    },
                                    timeout=60
                                )
                                if image_data.status_code == 200:
                                    images.append(image_data.content)
                                    found_images = True
                                    print(f"[ComfyClient] OK Imagen ({len(image_data.content)} bytes)")
                            
                            # ¡IMPORTANTE! Salir inmediatamente si encontramos imágenes
                            if found_images:
                                print(f"[ComfyClient] Saliendo del bucle - imagenes encontradas")
                                break
                        
                        if found_images:
                            break
                    
                    # Verificar estado sin depender del endpoint de job/status
                    has_images = len(images) > 0
                    if "status" in history[prompt_id] and history[prompt_id]["status"] in ("completed", "success"):
                        print(f"[ComfyClient] ¡Prompt completado!")
                        break
                    elif has_images:
                        break
                    elif history[prompt_id].get("status") == "failed":
                        error_msg = history[prompt_id].get("error", "Error desconocido")
                        raise Exception(f"Generacion fallida: {error_msg}")
            
            if found_images:
                print(f"[ComfyClient] Saliendo del bucle - imagenes encontradas")
                break
            
            # Esperar antes del siguiente intento
            time.sleep(0.5)
            
        if attempts >= max_attempts:
            print(f"[ComfyClient] Timeout esperando imagenes despues de {attempts} intentos")
        
        print(f"[ComfyClient] OK {len(images)} imagen(es) listas")
        return images
    
    def get_videos(self, prompt_id: str, node_id: str = "*") -> list:
        """Obtiene videos de un prompt (busca en 'gifs' y 'images')"""
        videos = []
        
        print(f"[ComfyClient] get_videos: {prompt_id[:8]}...")
        
        max_attempts = 1200  # Maximo 600 segundos esperando (10 minutos)
        attempts = 0
        found_videos = False
        nan_detected = False
        
        while attempts < max_attempts:
            attempts += 1
            
            try:
                history = self.get_history(prompt_id)
            except Exception as e:
                print(f"[ComfyClient] Error getting history: {e}")
                time.sleep(0.5)
                continue
            
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                status = history[prompt_id].get("status", {})
                
                # Verificar si hay errores NaN en los outputs
                for nid, node_output in outputs.items():
                    if "errors" in node_output:
                        for error in node_output["errors"]:
                            error_msg = str(error.get("message", ""))
                            if "nan" in error_msg.lower() or "NaN" in error_msg:
                                print(f"[ComfyClient] ERROR NaN detectado en nodo {nid}: {error_msg}")
                                nan_detected = True
                    # Verificar si hay datos inválidos en las imágenes
                    if "images" in node_output:
                        for img in node_output["images"]:
                            if img.get("filename", "").lower().endswith("nan"):
                                print(f"[ComfyClient] ERROR: Archivo con nombre NaN detectado")
                                nan_detected = True
                
                if nan_detected:
                    raise Exception("Error: El modelo generó valores NaN. Esto suele ocurrir por incompatibilidad con el VAE o el modelo. Prueba con otro VAE o modelo.")
                
                # Mostrar progreso cada 20 intentos
                if attempts % 20 == 0:
                    status_str = status.get("status", "unknown") if isinstance(status, dict) else str(status)
                    print(f"[ComfyClient] Esperando video... intento {attempts}, status={status_str}")
                
                if outputs:
                    print(f"[ComfyClient] Outputs disponibles: {list(outputs.keys())}")
                    
                    # Mostrar detalles de cada output para debug
                    for nid, node_output in outputs.items():
                        print(f"[ComfyClient]   Nodo {nid}: {list(node_output.keys())}")
                        if "errors" in node_output:
                            print(f"[ComfyClient]     ERRORS: {node_output['errors']}")
                    
                    # Buscar videos en todos los nodos (pueden estar en 'gifs' o 'images')
                    for nid, node_output in outputs.items():
                        # ComfyUI guarda videos en 'gifs' o a veces en 'images'
                        for key in ["gifs", "images"]:
                            if key in node_output:
                                for item in node_output[key]:
                                    filename = item.get("filename", "")
                                    # Verificar si es un archivo de video por extensión
                                    if any(filename.lower().endswith(ext) for ext in [".mp4", ".webm", ".mov", ".gif", ".avi"]):
                                        print(f"[ComfyClient] ¡VIDEO ENCONTRADO en nodo {nid} ({key}): {filename}!")
                                        video_data = requests.get(
                                            f"{self.base_url}/view",
                                            params={
                                                "filename": item["filename"],
                                                "subfolder": item.get("subfolder", ""),
                                                "type": item.get("type", "output")
                                            },
                                            timeout=120
                                        )
                                        if video_data.status_code == 200:
                                            videos.append(video_data.content)
                                            found_videos = True
                                            print(f"[ComfyClient] OK Video ({len(video_data.content)} bytes)")
                                        else:
                                            print(f"[ComfyClient] Error descargando video: {video_data.status_code}")
                                
                                if found_videos:
                                    break
                        
                        if found_videos:
                            break
                    
                    # Verificar estado
                    if "status" in history[prompt_id] and history[prompt_id]["status"] in ("completed", "success"):
                        print(f"[ComfyClient] ¡Prompt completado!")
                        break
                    elif found_videos:
                        break
                    elif history[prompt_id].get("status") == "failed":
                        error_msg = history[prompt_id].get("error", "Error desconocido")
                        raise Exception(f"Generacion fallida: {error_msg}")
            
            if found_videos:
                break
            
            # Esperar antes del siguiente intento
            time.sleep(0.5)
            
        if attempts >= max_attempts:
            print(f"[ComfyClient] Timeout esperando videos despues de {attempts} intentos")
        
        print(f"[ComfyClient] OK {len(videos)} video(s) listos")
        return videos
    
    def upload_image(self, image_path: str) -> Optional[str]:
        """Sube una imagen a ComfyUI"""
        if not os.path.exists(image_path):
            return None
        
        try:
            with open(image_path, "rb") as f:
                files = {"image": (os.path.basename(image_path), f, "image/png")}
                response = requests.post(
                    f"{self.base_url}/upload/image",
                    files=files,
                    data={"overwrite": "true"},
                    timeout=60
                )
            
            if response.status_code == 200:
                return response.json().get("name")
            return None
        except Exception as e:
            print(f"[ComfyClient] Error subiendo imagen: {e}")
            return None
    
    def upload_images_batch(self, image_paths: list) -> Optional[list]:
        """Sube multiples imagenes"""
        uploaded = []
        for path in image_paths:
            name = self.upload_image(path)
            if name:
                uploaded.append(name)
            else:
                print(f"[ComfyClient] Error subiendo: {path}")
        return uploaded if uploaded else None
    
    def get_checkpoints(self) -> list:
        """Obtiene lista de checkpoints disponibles"""
        try:
            response = requests.get(f"{self.base_url}/object_info/CheckpointLoaderSimple", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "CheckpointLoaderSimple" in data:
                    node = data["CheckpointLoaderSimple"]
                    if "input" in node and "required" in node["input"]:
                        ckpt_list = node["input"]["required"].get("ckpt_name")
                        if ckpt_list and len(ckpt_list) > 0 and len(ckpt_list[0]) > 0:
                            return ckpt_list[0]
            return []
        except Exception as e:
            print(f"[ComfyClient] Error obteniendo checkpoints: {e}")
            return []
    
    def get_models(self) -> list:
        """Alias de get_checkpoints para compatibilidad"""
        return self.get_checkpoints()

    def get_controlnet_models(self) -> list:
        """Obtiene lista de ControlNet models disponibles"""
        try:
            response = requests.get(f"{self.base_url}/object_info/ControlNetLoader", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "ControlNetLoader" in data:
                    node = data["ControlNetLoader"]
                    if "input" in node and "required" in node["input"]:
                        ckpt_list = node["input"]["required"].get("control_net_name")
                        if ckpt_list and len(ckpt_list) > 0 and len(ckpt_list[0]) > 0:
                            return ckpt_list[0]
            return []
        except Exception as e:
            print(f"[ComfyClient] Error obteniendo ControlNet models: {e}")
            return []

    def get_status(self, prompt_id: str) -> Dict:
        """Obtiene estado de un prompt"""
        try:
            response = requests.get(f"{self.base_url}/job/status/{prompt_id}", timeout=30)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}

    def get_object_info(self) -> Dict:
        """Obtiene información de nodos/modelos de ComfyUI"""
        try:
            response = requests.get(f"{self.base_url}/object_info", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"[ComfyClient] Error getting object_info: {e}")
        return {}

    def queue_prompt(self, workflow: Dict) -> Tuple[str, bool, str]:
        """Encola un prompt y retorna el ID del prompt"""
        try:
            payload = {"prompt": workflow, "client_id": self.client_id}
            response = requests.post(
                f"{self.base_url}/prompt",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                prompt_id = data.get("prompt_id")
                if prompt_id:
                    return prompt_id, True, ""
                else:
                    return "", False, "No se recibió prompt_id"
            elif response.status_code == 400:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Error de validacion")
                
                # Parsear detalles del error
                node_errors = error_data.get("node_errors", {})
                details = []
                for node_id, errors in node_errors.items():
                    for error in errors.get("errors", []):
                        error_type = error.get("type", "")
                        error_message = error.get("message", "")
                        extra_info = error.get("extra_info", {})
                        if extra_info:
                            received_value = extra_info.get("received_value", "")
                            if received_value:
                                details.append(f"[Nodo {node_id}] {error_type}: {error_message} (valor: {received_value})")
                            else:
                                details.append(f"[Nodo {node_id}] {error_type}: {error_message}")
                        else:
                            details.append(f"[Nodo {node_id}] {error_message}")
                
                full_error = f"{error_msg}: {', '.join(details)}" if details else error_msg
                
                # Agregar información de ayuda basada en el error
                help_info = self._get_error_help(full_error)
                if help_info:
                    full_error = f"{full_error}\n\n💡 {help_info}"
                
                return "", False, full_error
            else:
                return "", False, f"HTTP {response.status_code}: {response.text[:200]}"
                
        except requests.exceptions.ConnectionError:
            return "", False, "No se puede conectar a ComfyUI. ¿Esta corriendo?"
        except Exception as e:
            return "", False, f"Error: {str(e)}"
    
    def _get_error_help(self, error_msg: str) -> str:
        """Proporciona ayuda basada en el tipo de error"""
        error_lower = error_msg.lower()
        
        # Errores comunes y su ayuda
        if "SVD_img2vid_Conditioning" in error_msg:
            return "Instala el nodo 'ComfyUI-SVD-img2vid' para usar SVD Turbo. Busca en GitHub."
        elif "LTXVideo" in error_msg:
            return "Instala 'ComfyUI-LTXVideo' para usar LTX Video. Busca en GitHub."
        elif "UNETLoader" in error_msg and "not found" in error_lower:
            return "El modelo UNET no se encuentra. Verifica que el modelo esté en ui/tob/ComfyUI/models/diffusion_models/"
        elif "VAELoader" in error_msg and "not found" in error_lower:
            return "El VAE no se encuentra. Verifica que esté en ui/tob/ComfyUI/models/vae/"
        elif "clip_name" in error_lower or "clip vision" in error_lower:
            return "Falta el modelo CLIP Vision. Descarga 'open_clip_pytorch_model.bin' en ui/tob/ComfyUI/models/clip_vision/"
        elif "class_type" in error_lower:
            return "Un nodo del workflow no existe en ComfyUI. ¿Faltan nodos personalizados instalados?"
        elif "was not connected" in error_lower or "connection" in error_lower:
            return "ComfyUI no está respondiendo. Asegúrate de que está ejecutándose en el puerto 8188."
        
        return ""
    
    def generate_video(
        self,
        image_path: str,
        workflow: Dict,
        output_path: str
    ) -> Tuple[bool, Any]:
        """Genera video desde imagen con workflow.
        
        El workflow debe tener un nodo LoadImage con el nombre de archivo ya configurado.
        Este método solo encola el workflow y guarda el resultado.
        """
        try:
            # Verificar que la imagen existe
            if not os.path.exists(image_path):
                return False, f"Imagen no encontrada: {image_path}"
            
            print(f"[ComfyClient] ========== INICIANDO GENERACION ==========")
            print(f"[ComfyClient] Workflow keys: {list(workflow.keys())}")
            
            # El workflow ya debe tener la imagen configurada (cargada previamente)
            # Solo encolar y esperar resultado
            prompt_id, success, error = self.queue_prompt(workflow)
            
            print(f"[ComfyClient] queue_prompt result: success={success}, prompt_id={prompt_id[:8] if prompt_id else 'None'}..., error={error}")
            
            if not success:
                return False, error
            
            print(f"[ComfyClient] Generando video... ID: {prompt_id[:8]}...")
            time.sleep(2)
            
            # Esperar más tiempo para videos (hasta 10 minutos)
            max_wait = 600  # 10 minutos
            waited = 0
            videos = []
            
            print(f"[ComfyClient] Esperando hasta {max_wait}s por el video...")
            
            while waited < max_wait:
                videos = self.get_videos(prompt_id, "*")
                if videos:
                    print(f"[ComfyClient] Videos encontrados: {len(videos)}")
                    break
                time.sleep(5)
                waited += 5
                print(f"[ComfyClient] Esperando video... {waited}s / {max_wait}s", end='\r')
            
            print()  # Nueva línea después del progress
            
            if videos:
                print(f"[ComfyClient] Guardando video de {len(videos[0])} bytes en: {output_path}")
                with open(output_path, "wb") as f:
                    f.write(videos[0])
                print(f"[ComfyClient] Video guardado: {output_path}")
                return True, output_path
            else:
                # Intentar obtener el historial para ver qué pasó
                try:
                    history = self.get_history(prompt_id)
                    print(f"[ComfyClient] Historial final: {history}")
                except:
                    pass
                return False, "Timeout: No se generó el video en el tiempo esperado"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def generate_video_batch(
        self,
        image_paths: list,
        workflow_template: Dict
    ) -> Tuple[bool, Any]:
        """Genera video para multiples imagenes"""
        uploaded = self.upload_images_batch(image_paths)
        if not uploaded:
            return False, "Error subiendo imagenes"
        
        results = []
        for idx, image_filename in enumerate(uploaded):
            workflow = workflow_template.copy()
            workflow["6"] = {
                "inputs": {
                    "image": image_filename,
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {
                    "title": "LoadImage"
                }
            }
            
            prompt_id, success, error = self.queue_prompt(workflow)
            if success:
                images = self.get_images(prompt_id, "*")
                if images:
                    results.append(images[0])
                else:
                    results.append(None)
            else:
                results.append(None)
        
        return True, results


def check_comfy_available() -> bool:
    """Check if ComfyUI is available"""
    try:
        url = get_comfyui_url()
        response = requests.get(f"{url}/system_stats", timeout=5)
        return response.status_code == 200
    except:
        return False


def disable_safety_checker() -> bool:
    """Intenta desactivar el safety checker via API de ComfyUI"""
    try:
        url = get_comfyui_url()
        
        # Intentar endpoint de configuracion
        try:
            response = requests.post(
                f"{url}/config",
                json={"disable_safety_checker": True},
                timeout=5
            )
            if response.status_code == 200:
                print("[ComfyClient] Safety checker desactivado via API")
                return True
        except:
            pass
        
        # Intentar endpoint de manager
        try:
            response = requests.get(f"{url}/manager/disable-safety", timeout=5)
            if response.status_code == 200:
                print("[ComfyClient] Safety checker desactivado via Manager")
                return True
        except:
            pass
        
        print("[ComfyClient] No se pudo desactivar safety checker via API")
        return False
    except Exception as e:
        print(f"[ComfyClient] Error desactivando safety checker: {e}")
        return False


def get_comfy_version() -> str:
    """Get ComfyUI version"""
    try:
        url = get_comfyui_url()
        response = requests.get(f"{url}/system_stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("version", "Unknown")
    except:
        pass
    return "Unknown"
