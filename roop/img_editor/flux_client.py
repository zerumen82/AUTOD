# -*- coding: utf-8 -*-
"""
FluxClient - Cliente con Fallback a Stable Diffusion
"""

import os
import sys
import torch
import threading
import logging
from typing import Optional, Tuple
from PIL import Image
from dataclasses import dataclass

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Token HF por defecto (REEMPLAZAR CON TU TOKEN)
DEFAULT_HF_TOKEN = ""

# Modelos
FLUX_MODEL = "black-forest-labs/FLUX.1-fill-dev"
SD_MODEL = "runwayml/stable-diffusion-v1-5"

try:
    from diffusers import FluxFillPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
    DIFFUSERS_AVAILABLE = True
    DIFFUSERS_ERROR = None
except ImportError as e:
    DIFFUSERS_AVAILABLE = False
    DIFFUSERS_ERROR = str(e)


@dataclass
class GenerationResult:
    image: Image.Image
    seed: Optional[int] = None
    mode: str = "unknown"
    time_taken: float = 0.0


class FluxClient:
    """Cliente con fallback automático a Stable Diffusion si FLUX falla."""
    
    def __init__(self):
        self.pipe_flux: Optional[FluxFillPipeline] = None
        self.pipe_sd_img2img: Optional[StableDiffusionImg2ImgPipeline] = None
        self.pipe_sd_inpaint: Optional[StableDiffusionInpaintPipeline] = None
        self.current_model: str = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._lock = threading.Lock()
        self._is_loaded = False
        self._is_loading = False
        
    def is_available(self) -> bool:
        return DIFFUSERS_AVAILABLE
    
    def get_installation_help(self) -> str:
        if DIFFUSERS_AVAILABLE:
            return ""
        return "Instala: pip install diffusers transformers torch pillow"
    
    def _log(self, msg: str, level: str = "INFO"):
        prefix = {"INFO": "[INFO]", "WARNING": "[WARNING]", "ERROR": "[ERROR]", "SUCCESS": "[SUCCESS]"}.get(level, "[INFO]")
        print(f"{prefix} FluxClient: {msg}")
    
    def load(self, progress_callback=None, hf_token: Optional[str] = None, use_sd_first: bool = False) -> Tuple[bool, str]:
        """
        Carga FLUX Fill Pipeline como modelo principal.
        
        Args:
            use_sd_first: Si True, intenta cargar SD primero (más rápido). Si False, FLUX primero.
        """
        with self._lock:
            if self._is_loaded:
                return True, f"Modelo ya cargado ({self.current_model})"
            if self._is_loading:
                return False, "Modelo ya esta cargandose"
            self._is_loading = True
        
        try:
            if not DIFFUSERS_AVAILABLE:
                error_msg = f"Diffusers no disponible: {DIFFUSERS_ERROR}"
                self._log(error_msg, "ERROR")
                return False, error_msg
            
            # Estrategia de carga: FLUX primero (mejor calidad), SD como fallback
            models_to_try = []
            if use_sd_first:
                models_to_try = ["sd_img2img", "flux"]
            else:
                models_to_try = ["flux", "sd_img2img"]
            
            for model_type in models_to_try:
                try:
                    if model_type == "flux":
                        self._log("Cargando FLUX Fill Pipeline (calidad superior)...")
                        self.pipe_flux = FluxFillPipeline.from_pretrained(
                            FLUX_MODEL,
                            torch_dtype=self.dtype,
                            low_cpu_mem_usage=True,
                        )
                        self.pipe_flux = self.pipe_flux.to(self.device)
                        self._configure_optimizations(self.pipe_flux)
                        self.current_model = "FLUX.1-fill-dev"
                        self._is_loaded = True
                        self._is_loading = False
                        self._log("✅ FLUX Fill Pipeline cargado!", "SUCCESS")
                        return True, "FLUX.1-fill-dev"
                        
                    elif model_type == "sd_img2img":
                        self._log("Cargando Stable Diffusion v1.5 Img2Img (rápido)...")
                        self.pipe_sd_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                            SD_MODEL,
                            torch_dtype=self.dtype,
                            low_cpu_mem_usage=True,
                        )
                        self.pipe_sd_img2img = self.pipe_sd_img2img.to(self.device)
                        self._configure_optimizations_sd(self.pipe_sd_img2img)
                        self.current_model = "Stable Diffusion v1.5"
                        self._is_loaded = True
                        self._is_loading = False
                        self._log("✅ Stable Diffusion Img2Img cargado!", "SUCCESS")
                        return True, "Stable Diffusion v1.5"
                        
                except Exception as model_error:
                    self._log(f"❌ Error cargando {model_type}: {model_error}", "WARNING")
                    continue
            
            # Si llegamos aquí, todos los modelos fallaron
            self._is_loading = False
            return False, "No se pudo cargar ningún modelo (FLUX ni SD)"
                    
        except Exception as e:
            self._is_loading = False
            self._log(f"❌ Error general en load(): {e}", "ERROR")
            import traceback
            self._log(traceback.format_exc(), "ERROR")
            return False, str(e)
    
    def _configure_optimizations(self, pipe):
        """Configura optimizaciones para FLUX (8GB VRAM)."""
        if pipe is None:
            return
        
        self._log("Aplicando optimizaciones VRAM para FLUX...")
        
        # 1. Tiled VAE - procesa imagen en tiles
        try:
            pipe.enable_vae_tiling()
            self._log("  ✓ VAE tiling activado")
        except Exception as e:
            self._log(f"  ✗ VAE tiling no disponible: {e}", "WARNING")
        
        # 2. Attention slicing - reduce pico de VRAM
        try:
            pipe.enable_attention_slicing()
            self._log("  ✓ Attention slicing activado")
        except Exception as e:
            self._log(f"  ✗ Attention slicing no disponible: {e}", "WARNING")
        
        # 3. CPU offload si VRAM es muy limitada (< 6GB)
        try:
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if vram_gb < 6:
                    pipe.enable_sequential_cpu_offload()
                    self._log("  ✓ CPU offload activado (VRAM < 6GB)")
                else:
                    self._log(f"  ℹ️ CPU offload no necesario (VRAM: {vram_gb:.1f}GB)")
        except Exception as e:
            self._log(f"  ✗ CPU offload no disponible: {e}", "WARNING")
        
        # 4. Desactivar safety checker (sin censura)
        try:
            pipe.safety_checker = None
            if hasattr(pipe, 'feature_extractor'):
                pipe.feature_extractor = None
            self._log("  ✓ Safety checker desactivado")
        except Exception as e:
            self._log(f"  ✗ No se pudo desactivar safety checker: {e}", "WARNING")
    
    def _configure_optimizations_sd(self, pipe):
        """Configura optimizaciones para SD (4-6GB VRAM)."""
        if pipe is None:
            return
        
        self._log("Aplicando optimizaciones VRAM para SD...")
        
        # 1. Tiled VAE
        try:
            pipe.vae.enable_tiling()
            self._log("  ✓ VAE tiling activado")
        except Exception as e:
            self._log(f"  ✗ VAE tiling no disponible: {e}", "WARNING")
        
        # 2. Attention slicing
        try:
            pipe.enable_attention_slicing()
            self._log("  ✓ Attention slicing activado")
        except Exception as e:
            self._log(f"  ✗ Attention slicing no disponible: {e}", "WARNING")
        
        # 3. Desactivar safety checker
        try:
            pipe.safety_checker = None
            self._log("  ✓ Safety checker desactivado")
        except Exception as e:
            self._log(f"  ✗ No se pudo desactivar safety checker: {e}", "WARNING")
    
    def unload(self):
        with self._lock:
            if self.pipe_flux is not None:
                del self.pipe_flux
                self.pipe_flux = None
            if self.pipe_sd_img2img is not None:
                del self.pipe_sd_img2img
                self.pipe_sd_img2img = None
            if self.pipe_sd_inpaint is not None:
                del self.pipe_sd_inpaint
                self.pipe_sd_inpaint = None
            self._is_loaded = False
            self.current_model = ""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._log("Modelos descargados")
    
    def is_loaded(self) -> bool:
        return self._is_loaded and (
            self.pipe_flux is not None or 
            self.pipe_sd_img2img is not None or
            self.pipe_sd_inpaint is not None
        )
    
    def get_current_model(self) -> str:
        return self.current_model
    
    def generate_fill(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 8,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        seed: Optional[int] = None,
        use_cpu_fallback: bool = False,
    ) -> Tuple[Optional[GenerationResult], str]:
        """
        Genera imagen modificada usando el modelo cargado.
        
        Para SD: usa img2img (strength controla la modificación)
        Para FLUX: usa fill pipeline (strength se ignora, usa pasos y guidance)
        """
        if not self.is_loaded():
            return None, "Modelo no cargado"
        
        try:
            import time
            start_time = time.time()
            generator = None
            if seed is not None:
                generator = torch.Generator(self.device).manual_seed(seed)
            
            # Determinar dispositivo
            device = "cpu" if use_cpu_fallback else self.device
            
            if self.pipe_flux is not None:
                # FLUX Fill Pipeline (no usa strength)
                self._log("Generando con FLUX Fill...")
                pipe = self.pipe_flux.to(device)
                
                # FLUX Fill: pasar imagen directamente
                result = pipe(
                    prompt=prompt,
                    image=image,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
                mode = "FLUX-Fill"
                
            elif self.pipe_sd_img2img is not None:
                # SD Img2Img (usa strength)
                self._log("Generando con SD Img2Img...")
                pipe = self.pipe_sd_img2img.to(device)
                result = pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator,
                ).images[0]
                mode = "SD-Img2Img"
            else:
                return None, "No hay modelo cargado"
            
            time_taken = time.time() - start_time
            self._log(f"✅ {mode} completado en {time_taken:.1f}s")
            
            return GenerationResult(
                image=result,
                seed=seed,
                mode=mode,
                time_taken=time_taken
            ), f"{mode} OK en {time_taken:.1f}s"
            
        except Exception as e:
            error_msg = str(e)
            self._log(f"❌ Error en generate_fill: {error_msg}", "ERROR")
            
            # Si hay error de memoria CUDA, intentar con CPU
            if "CUDA out of memory" in error_msg and not use_cpu_fallback:
                self._log("🔄 CUDA out of memory, reintentando con CPU...", "WARNING")
                return self.generate_fill(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    seed=seed,
                    use_cpu_fallback=True,
                )
            
            import traceback
            self._log(traceback.format_exc(), "ERROR")
            return None, f"Error: {error_msg}"
    
    def generate_inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 8,
        guidance_scale: float = 7.5,
        strength: float = 1.0,
        seed: Optional[int] = None,
        use_cpu_fallback: bool = False,
    ) -> Tuple[Optional[GenerationResult], str]:
        """
        Genera inpainting en un area especifica (mask).
        
        Args:
            image: Imagen de entrada
            mask: Mascara (blanco = area a modificar, negro = preservar)
            prompt: Descripcion de cambios
            negative_prompt: Lo que NO queremos
            num_inference_steps: Pasos de inferencia
            guidance_scale: Adherencia al prompt
            strength: Intensidad del cambio
            seed: Semilla para reproducibilidad
            use_cpu_fallback: Si usar CPU si GPU falla
            
        Returns:
            (result_image, status_message)
        """
        if not self.is_loaded():
            return None, "Modelo no cargado"
        
        try:
            import time
            start_time = time.time()
            generator = None
            if seed is not None:
                generator = torch.Generator(self.device).manual_seed(seed)
            
            # Determinar dispositivo
            device = "cpu" if use_cpu_fallback else self.device
            
            # Cargar pipeline de inpaint si no existe
            if self.pipe_sd_inpaint is None:
                self._log("Cargando Stable Diffusion Inpaint...")
                self.pipe_sd_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                    SD_MODEL,
                    dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
                self.pipe_sd_inpaint = self.pipe_sd_inpaint.to(device)
                self._configure_optimizations_sd(self.pipe_sd_inpaint)
            
            pipe = self.pipe_sd_inpaint.to(device)
            
            # Redimensionar mask si es necesario
            mask_resized = mask.resize(image.size, Image.LANCZOS)
            
            result = pipe(
                image=image,
                mask_image=mask_resized,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            ).images[0]
            
            mode = "SD-Inpaint"
            time_taken = time.time() - start_time
            return GenerationResult(
                image=result, 
                seed=seed, 
                mode=mode, 
                time_taken=time_taken
            ), f"{mode} OK en {time_taken:.1f}s"
            
        except Exception as e:
            error_msg = str(e)
            if "CUDA out of memory" in error_msg and not use_cpu_fallback:
                self._log(" CUDA out of memory, reintentando con CPU...", "WARNING")
                return self.generate_inpaint(
                    image=image,
                    mask=mask,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    seed=seed,
                    use_cpu_fallback=True,
                )
            return None, f"Error: {error_msg}"
    
    def get_vram_info(self) -> dict:
        if not torch.cuda.is_available():
            return {"device": "CPU", "allocated": 0, "total": 0}
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }
    
    def __del__(self):
        try:
            self.unload()
        except:
            pass


_flux_client: Optional[FluxClient] = None


def get_flux_client() -> FluxClient:
    global _flux_client
    if _flux_client is None:
        _flux_client = FluxClient()
    return _flux_client


def is_flux_available() -> bool:
    """Verifica si FLUX está disponible (diffusers + modelo descargado)"""
    client = get_flux_client()
    if not client.is_available():
        return False
    
    # Verificar si el modelo FLUX está en cache local
    try:
        from huggingface_hub import snapshot_cache
        import os
        
        model_id = FLUX_MODEL
        cache_path = snapshot_cache(model_id)
        if os.path.exists(cache_path):
            # Verificar que existan archivos clave
            required_files = [
                "model_index.json",
                "diffusion_pytorch_model.safetensors"  # o .bin
            ]
            for f in required_files:
                if os.path.exists(os.path.join(cache_path, f)):
                    return True
        return False
    except Exception as e:
        print(f"[FluxClient] Error verificando cache: {e}")
        return False


def is_flux_loaded() -> bool:
    """Verifica si FLUX está cargado actualmente"""
    return get_flux_client().is_loaded()
