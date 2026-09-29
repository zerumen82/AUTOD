"""Verify timeout cleanup targets only the slow prompt before a fallback."""

from unittest.mock import patch

from roop.img_editor.comfy_progress import stop_comfy_prompt, wait_for_comfy_image


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def main():
    slow_id = "qwen-slow"
    next_id = "longcat-next"
    state = {"running": slow_id, "pending": next_id}
    posts = []

    def get(url, **_kwargs):
        assert url.endswith("/queue"), url
        return Response({
            "queue_running": [[0, state["running"]]] if state["running"] else [],
            "queue_pending": [[1, state["pending"]]] if state["pending"] else [],
        })

    def post(url, json=None, **_kwargs):
        posts.append((url.rsplit("/", 1)[-1], json))
        if url.endswith("/interrupt"):
            assert json == {"prompt_id": slow_id}, json
            state["running"] = None
        return Response({})

    with patch("roop.img_editor.comfy_progress.requests.get", side_effect=get), \
         patch("roop.img_editor.comfy_progress.requests.post", side_effect=post), \
         patch("roop.img_editor.comfy_progress.time.sleep"):
        assert stop_comfy_prompt("http://comfy", slow_id, timeout=5)
    assert posts == [("interrupt", {"prompt_id": slow_id})], posts
    assert state["pending"] == next_id

    # A running task that ignores interruption must not be reported as stopped.
    state["running"] = slow_id
    with patch("roop.img_editor.comfy_progress.requests.get", side_effect=get), \
         patch("roop.img_editor.comfy_progress.requests.post", return_value=Response({})), \
         patch("roop.img_editor.comfy_progress.stop_comfy_prompt", return_value=False):
        meta, note = wait_for_comfy_image("http://comfy", slow_id, timeout=-1)
    assert meta is None and "sigue ocupado" in note, note
    print("OK: timeout interrupts by ID, preserves other work, blocks unsafe fallback")


if __name__ == "__main__":
    main()
