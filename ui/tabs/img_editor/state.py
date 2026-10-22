import threading
import copy


class SelectionSession:
    """Mutable session state; in-flight analyses observe invalidation immediately."""

    def __init__(self):
        self._lock = threading.RLock()
        self.revision = 0
        self.analysis = None
        self.selected_ids = []
        self.current_fingerprint = None
        self.image_bound = False

    def __getstate__(self):
        # Exclude the lock for pickling (Gradio may serialize state)
        return {k: v for k, v in self.__dict__.items() if k != '_lock'}

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)
        self._lock = threading.RLock()

    def __deepcopy__(self, memo):
        result = type(self)()
        memo[id(self)] = result
        with self._lock:
            result.revision = self.revision
            result.analysis = copy.deepcopy(self.analysis, memo)
            result.selected_ids = list(self.selected_ids)
            result.current_fingerprint = self.current_fingerprint
            result.image_bound = self.image_bound
        return result

    @property
    def lock(self):
        return self._lock

    def invalidate(self):
        with self._lock:
            self.revision += 1
            self.analysis = None
            self.selected_ids = []
            return self.revision

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self.analysis), list(self.selected_ids)

    def replace_image(self, fingerprint):
        with self._lock:
            self.invalidate()
            self.current_fingerprint = fingerprint
            # Removing an image also binds the empty state: a delayed request
            # for its former photo must not initialize this session again.
            self.image_bound = True

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
