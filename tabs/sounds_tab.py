import os
import gradio as gr
import numpy as np
import torch
from TTS.api import TTS
from TTS.utils.manage import ModelManager
import soundfile as sf
from TTS.utils.radam import RAdam
from collections import defaultdict

# Add RAdam and defaultdict to the safe globals list
torch.serialization.add_safe_globals([RAdam, defaultdict])
OUTPUT_DIR = "out-voice"

def open_output_folder():
    os.startfile(OUTPUT_DIR)

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

def loopInModels(models, path):
        returner = []
        for model in models:
            if model.startswith(path):
                returner.append(model)
        return returner

def getModels(lenguaje):
    path = "tts_models"
    tts = TTS()
    modelos = tts.list_models()  # This returns the list of models directly
    abLen = lenguajeJoiner(lenguaje)
    pith = path + "/" + abLen
    returner = loopInModels(modelos, pith)
    return returner

def update_model_dropdown(lenguaje):
    return gr.update(choices=getModels(lenguaje))

def convert_audio_to_16bit(audio_path):
    data, samplerate = sf.read(audio_path)
    if data.dtype != np.int16:
        data = (data * 32767).astype(np.int16)
        sf.write(audio_path, data, samplerate)
    return audio_path

def inference(text, lenguaje, modelo, mic_audio, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    abLen = lenguajeJoiner(lenguaje)
    tts = TTS(modelo.value).to(device)  # Convertir modelo a cadena
    mic_audio = convert_audio_to_16bit(mic_audio)
    name = os.path.basename(mic_audio)
    output_path = os.path.join(OUTPUT_DIR, "output.wav")
    if lenguaje == "Multilenguaje":
        wav = tts.tts(text=text, speaker_wav=mic_audio, language=abLen)
        tts.tts_to_file(text=text, speaker_wav=mic_audio, language=abLen, file_path=os.path.join(OUTPUT_DIR, name + ".wav"))
    else:
        tts.tts_to_file(text=text, speaker_wav=mic_audio, file_path=os.path.join(OUTPUT_DIR, name + ".wav"))
    return output_path

def sounds_tab():
    with gr.Tab("🔊 Clonado de Voz"):
        text = gr.Textbox(lines=4, label="Texto:")
        lenguaje = gr.Radio(
            ["Español", "Inglés", "Alemán", "Francés", "Italiano", "Portugués", "Multilenguaje"],
            label="Elige Idioma de la conversión",
            type="value",
            value="Español"
        )
        modelo = gr.Dropdown(choices=getModels(lenguaje.value), label="Modelo de voz", interactive=True)
        mic_audio = gr.Audio(
            label="Sube o graba la voz:",
            type="filepath",
        )

        lenguaje.change(fn=update_model_dropdown, inputs=lenguaje, outputs=modelo)
        output_dir = OUTPUT_DIR

        seed = gr.Number(value=6335544, precision=0, label="Semilla:")

        selected_voice = gr.Audio(label="Sample de la voz subida")
        gr.Button("Convertir").click(fn=inference, inputs=[text, lenguaje, modelo, mic_audio, seed], outputs=selected_voice)
