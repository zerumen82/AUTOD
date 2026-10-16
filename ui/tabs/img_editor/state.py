import threading

is_generating = False
# Marca de tiempo de arranque de la generación activa. Si is_generating sigue
# True pero este valor lleva muchísimo rato, el guard está clavado y el
# rechazo lo comunica con el tiempo transcurrido en vez de un "no" mudo.
generation_started_at = None
_generation_lock = threading.Lock()

last_input_image = None
last_result_image = None
history = []

autoregressive_mode = False
