import threading

is_generating = False
_generation_lock = threading.Lock()

last_input_image = None
last_result_image = None
history = []

autoregressive_mode = False
