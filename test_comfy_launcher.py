import sys
sys.path.append('d:/PROJECTS/AUTOAUTO')

import time
import threading
from ui.tabs.comfy_launcher import start, stop, get_last_url, get_last_port
import requests

print('=== Probando Launcher de ComfyUI ===')

# Detener cualquier instancia anterior de ComfyUI
print('1. Deteniendo instancias anteriores de ComfyUI...')
stop()
time.sleep(2)

# Iniciar ComfyUI
print('2. Iniciando ComfyUI...')
thread = start()
if thread is not None:
    print('   Thread de inicio creado')
else:
    print('   Error al crear el thread de inicio')

# Esperar a que se inicie
print('3. Esperando a que ComfyUI se inicie...')
max_wait = 120  # 2 minutos máximo
waited = 0

while waited < max_wait:
    time.sleep(5)
    waited += 5
    url = get_last_url()
    port = get_last_port()
    
    if url and port:
        print(f'   URL: {url}, Puerto: {port}')
        
        # Intentar conectarse
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print('✅ ComfyUI se ha iniciado correctamente')
                print(f'   Código de estado: {response.status_code}')
                print(f'   Contenido (primeros 200 chars): {response.text[:200]}')
                break
            else:
                print(f'⚠️  Respuesta del servidor: {response.status_code}')
        except Exception as e:
            print(f'   Error al conectar: {e}')
    else:
        print('   URL y puerto aún no disponibles')
        
    if waited % 30 == 0:
        print(f'   Esperado {waited} segundos...')

if waited >= max_wait:
    print('❌ Timeout: ComfyUI no se inició en el tiempo esperado')

print('=== Fin de la prueba ===')
