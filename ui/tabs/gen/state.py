import threading

is_generating = False
_generation_lock = threading.Lock()
last_result_image = None
