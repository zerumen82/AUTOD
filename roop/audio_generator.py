
import os
import torch
import soundfile as sf
import numpy as np

# --- PATCH PARA COMPATIBILIDAD CON TRANSFORMERS NUEVOS ---
try:
    from TTS.tts.layers.xtts.gpt import GPT2InferenceModel
    from transformers.generation import GenerationMixin
    
    # Verificar si ya hereda de GenerationMixin
    if GenerationMixin not in GPT2InferenceModel.__bases__:
        print("Patching GPT2InferenceModel to inherit from GenerationMixin (Fix for transformers >= 4.50)...")
        # Añadir GenerationMixin a las bases
        GPT2InferenceModel.__bases__ = (GenerationMixin,) + GPT2InferenceModel.__bases__
except Exception as e:
    # No fallar si esto no funciona, solo advertir (puede que no sea XTTS o la versión sea diferente)
    print(f"Could not patch GPT2InferenceModel: {e}")
# ---------------------------------------------------------

OUTPUT_DIR = "out-voice"

def lenguajeJoiner(lenguaje):
    if lenguaje == "Español":
        return "es"
    elif lenguaje == "Inglés":
        return "en"
    elif lenguaje == "Alemán":
        return "de"
    elif lenguaje == "Francés":
        return "fr"
    elif lenguaje == "Italiano":
        return "it"
    elif lenguaje == "Portugués":
        return "pt"
    elif lenguaje == "Multilenguaje":
        return "multilingual"
    else:
        return "es"

def get_models(lenguaje):
    try:
        from TTS.api import TTS
    except ImportError:
        print("[ERROR] TTS library not installed")
        return ["tts_models/multilingual/multi-dataset/xtts_v2"]
    
    lang_code = lenguajeJoiner(lenguaje)
    
    # Instanciar TTS sin cargar modelo para listar (si es posible) o usar manejo de excepciones
    try:
        # TTS() sin argumentos puede intentar cargar un modelo por defecto, mejor usar list_models directamente si es estático
        # Pero TTS.list_models() es un método de instancia en algunas versiones.
        # Asumiremos que instanciar TTS() es seguro o ligero, o capturamos error.
        tts = TTS() 
        all_models = tts.manager.list_models()
    except Exception:
        # Fallback si falla la inicialización vacía
        return ["tts_models/multilingual/multi-dataset/xtts_v2"]
    
    if lang_code == "multilingual":
        returner = [model for model in all_models if "multilingual" in model.lower()]
    else:
        lang_patterns = {
            "es": ["spanish", "es_"],
            "en": ["english", "en_"],
            "de": ["german", "de_"],
            "fr": ["french", "fr_"],
            "it": ["italian", "it_"],
            "pt": ["portuguese", "pt_"]
        }
        patterns = lang_patterns.get(lang_code, [lang_code])
        returner = [model for model in all_models 
                   if any(pattern in model.lower() for pattern in patterns)]
    
    return returner if returner else ["tts_models/multilingual/multi-dataset/xtts_v2"]

def generate_audio(text, lenguaje, model_name=None, speaker_wav=None, output_path=None):
    """Generate audio from text using TTS.
    
    Args:
        text: Text to convert to speech
        lenguaje: Language (Español, Inglés, etc.)
        model_name: TTS model name (optional)
        speaker_wav: Reference audio for voice cloning (optional)
        output_path: Output file path (optional)
    
    Returns:
        Path to generated audio file, or None if failed
    """
    # Check if TTS is available
    try:
        from TTS.api import TTS
    except ImportError as e:
        print(f"[ERROR] ERROR: TTS library not installed: {e}")
        print("💡 To enable audio generation, run:")
        print("   pip install TTS soundfile")
        print("   or: pip install -r requirements.txt")
        return None
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Obtener código de idioma correcto
    abLen = lenguajeJoiner(lenguaje)
    # Si devuelve "multilingual", forzar español por defecto para generación
    if abLen == "multilingual":
        abLen = "es"
    
    if model_name is None:
        # Default to a good multilingual model if none specified
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
    print(f"🎙️ Cargando modelo TTS: {model_name} en {device}")
    
    try:
        tts = TTS(model_name).to(device)
    except Exception as e:
        print(f"[ERROR] Error loading TTS model: {e}")
        print("💡 This might be due to:")
        print("   1. Missing TTS library (run: pip install TTS)")
        print("   2. Missing protobuf (run: pip install protobuf)")
        print("   3. Model download failed (check internet connection)")
        return None
    
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, f"generated_audio_{hash(text)}.wav")
        
    # Manejo de speaker_wav para modelos XTTS (requieren referencia de voz)
    if "xtts" in model_name.lower() and not speaker_wav:
        # Intentar buscar un wav existente en el directorio de salida para usar como referencia
        if os.path.exists(OUTPUT_DIR):
            wavs = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
            if wavs:
                speaker_wav = os.path.join(OUTPUT_DIR, wavs[0])
                print(f"⚠️ Usando audio de referencia encontrado: {speaker_wav}")
            else:
                # Si no hay audio, intentar usar un archivo por defecto si existe en assets
                default_ref = os.path.join(os.getcwd(), "assets", "ref_voice.wav")
                if os.path.exists(default_ref):
                    speaker_wav = default_ref
                    print(f"⚠️ Usando audio de referencia por defecto: {speaker_wav}")
                else:
                    print("⚠️ ADVERTENCIA: XTTS requiere un archivo de audio para clonar voz (speaker_wav).")
                    print("   Se intentará generar sin él, pero podría fallar.")

    # Preparar argumentos para tts_to_file
    kwargs = {}
    
    # 1. Manejo de Idioma
    if tts.is_multi_lingual:
        kwargs["language"] = abLen
        
    # 2. Manejo de Speaker (Voz)
    if speaker_wav and os.path.exists(speaker_wav):
        # Asegurar formato correcto del wav
        try:
            data, samplerate = sf.read(speaker_wav)
            if data.dtype != np.int16:
                data = (data * 32767).astype(np.int16)
                sf.write(speaker_wav, data, samplerate)
        except Exception as e:
            print(f"⚠️ Error procesando speaker_wav: {e}")
            
        kwargs["speaker_wav"] = speaker_wav
    elif tts.is_multi_speaker:
        # Si no hay wav pero el modelo tiene speakers predefinidos, usar el primero
        # Safely access speakers
        speakers = getattr(tts, "speakers", None)
        if speakers:
            kwargs["speaker"] = speakers[0]
    
    # Generar audio
    try:
        print(f"🔊 Generando audio en '{abLen}'...")
        tts.tts_to_file(text=text, file_path=output_path, **kwargs)
        return output_path
    except Exception as e:
        print(f"[ERROR] Error en tts_to_file: {e}")
        # Reintentar sin speaker_wav si falló y lo teníamos
        if "speaker_wav" in kwargs:
            print("🔄 Reintentando sin speaker_wav...")
            del kwargs["speaker_wav"]
            # Si es multispeaker, intentar asignar un speaker ID
            if tts.is_multi_speaker:
                speakers = getattr(tts, "speakers", None)
                if speakers:
                    kwargs["speaker"] = speakers[0]
            try:
                tts.tts_to_file(text=text, file_path=output_path, **kwargs)
                return output_path
            except Exception as e2:
                print(f"[ERROR] Falló reintento: {e2}")
                raise e
        raise e


def generate_sound(text, duration=5.0, steps=20, output_path=None):
    """
    Genera efectos de sonido o música usando AudioLDM (via diffusers).
    Requiere: pip install diffusers transformers scipy
    """
    try:
        from diffusers import AudioLDMPipeline
        import torch
    except ImportError:
        print("[ERROR] diffusers library not installed for AudioLDM.")
        print("💡 To enable ambient sound generation, run:")
        print("   pip install diffusers transformers scipy accelerate")
        return None

    if output_path is None:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, f"generated_sfx_{hash(text)}.wav")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"🎵 Cargando AudioLDM para: '{text}' en {device}...")
    
    try:
        # Usar modelo small para velocidad y compatibilidad (cvssp/audioldm-s-full-v2)
        repo_id = "cvssp/audioldm-s-full-v2"
        pipe = AudioLDMPipeline.from_pretrained(repo_id, dtype=dtype)
        pipe = pipe.to(device)
        
        # Optimización de memoria
        if device == "cuda":
            pipe.enable_attention_slicing()
        
        print("🔊 Generando audio ambiente/música...")
        audio = pipe(text, num_inference_steps=steps, audio_length_in_s=duration).audios[0]
        
        # Guardar con scipy
        import scipy.io.wavfile
        sample_rate = pipe.unet.config.sample_size if hasattr(pipe.unet.config, 'sample_size') else 16000
        # AudioLDM usually outputs 16khz
        scipy.io.wavfile.write(output_path, rate=16000, data=audio)
        
        print(f"✅ Audio ambiente guardado: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error generando AudioLDM: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

