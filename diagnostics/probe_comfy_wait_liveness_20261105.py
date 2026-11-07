"""CPU contracts for wait timing and progress; protocol doubles, no inference."""

import contextlib
import io
import unittest
from unittest.mock import patch

from roop.img_editor import comfy_progress as cp


class Clock:
    def __init__(self):
        self.now = 100.0

    def tick(self, amount=1.0):
        self.now += amount


class Response:
    status_code = 200

    def __init__(self, value):
        self.value = value

    def json(self):
        return self.value


class WaitLivenessContracts(unittest.TestCase):
    def run_wait(self, *, wall_jump=False, rounds=2, error=False):
        clock = Clock()
        order = []
        callbacks = []
        output = io.StringIO()
        count = 0
        ws = cp.WsProgress("http://test", "target", {"8": "KSampler"})

        def start():
            ws.handle_payload({"type": "executing", "data": {
                "prompt_id": "target", "node": "8",
            }})
            ws.handle_payload({"type": "progress", "data": {
                "prompt_id": "target", "value": 3, "max": 12,
            }})
            if error:
                ws.handle_payload({"type": "execution_error", "data": {
                    "prompt_id": "target", "exception_message": "CUDA out of memory",
                }})

        def check(*_args, **_kwargs):
            nonlocal count
            count += 1
            order.append("http")
            clock.tick(2)
            return {"done": True} if count >= rounds else {"running": True}

        def callback(prog):
            order.append("callback")
            callbacks.append(prog.copy())
            clock.tick(0.25)

        hist = {"target": {
            "status": {"status_str": "error" if error else "success", "completed": True,
                       "messages": [["execution_start", {"timestamp": 1000000}],
                                    ["execution_success", {"timestamp": 1008000}]]},
            "outputs": {"10": {"images": [{"filename": "real-protocol-output.png"}]}},
        }}
        settings = {"comfy_progress": {
            "heartbeat_interval_seconds": 5,
            "status_timeout_seconds": 10,
            "callback_interval_seconds": 0.8,
        }}
        def wall():
            return clock.now + (10000 if wall_jump and clock.now >= 102 else 0)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(cp.time, "monotonic", side_effect=lambda: clock.now))
            stack.enter_context(patch.object(cp.time, "time", side_effect=wall))
            stack.enter_context(patch.object(cp.time, "sleep", side_effect=clock.tick))
            stack.enter_context(patch.object(ws, "start", side_effect=start))
            stack.enter_context(patch.object(ws, "stop"))
            stack.enter_context(patch.object(cp, "WsProgress", return_value=ws))
            stack.enter_context(patch.object(cp, "check_comfy_progress", side_effect=check))
            stack.enter_context(patch.object(cp, "history_mean_seconds", return_value=0))
            stack.enter_context(patch.object(cp.requests, "get", return_value=Response(hist)))
            stack.enter_context(patch.object(cp, "get_generation_prompt_config",
                                            return_value=settings, create=True))
            stopped = stack.enter_context(patch.object(cp, "stop_comfy_prompt", return_value=True))
            stack.enter_context(contextlib.redirect_stdout(output))
            result = cp.wait_for_comfy_image("http://test", "target", timeout=100,
                                            progress_callback=callback)
        return result, callbacks, order, output.getvalue(), stopped

    def test_wall_clock_jump_does_not_cancel_valid_work(self):
        result, _, _, _, stopped = self.run_wait(wall_jump=True)
        self.assertIsNotNone(result[0])
        stopped.assert_not_called()

    def test_ws_progress_is_delivered_before_http_poll(self):
        _, callbacks, order, _, _ = self.run_wait()
        self.assertEqual(order[0], "callback")
        self.assertEqual(callbacks[0]["step"], 3)

    def test_heartbeat_keeps_real_step_without_claiming_health(self):
        _, _, _, output, _ = self.run_wait(rounds=7)
        self.assertGreaterEqual(output.count("Muestreando paso 3/12"), 3)
        self.assertIn("WS", output)
        self.assertNotIn("No está colgado", cp.build_generation_progress_html())
        self.assertNotIn("paso 4/12", output)

    def test_completion_reports_server_and_local_timing(self):
        _, callbacks, _, output, _ = self.run_wait()
        timing = callbacks[-1]["timing"]
        self.assertEqual(timing["server_seconds"], 8.0)
        self.assertGreaterEqual(timing["http_max_seconds"], 2)
        self.assertGreaterEqual(timing["callback_max_seconds"], 0.25)
        self.assertGreaterEqual(callbacks[-1]["elapsed"], 4)
        self.assertIn("monotónico", output)

    def test_error_ws_never_returns_partial_image_as_success(self):
        result, _, _, _, _ = self.run_wait(error=True, rounds=1)
        self.assertIsNone(result[0])
        self.assertIn("CUDA", result[1])

    def test_ws_node_durations_ignore_wall_clock(self):
        ws = cp.WsProgress("http://test", "target", {"8": "KSampler"})
        clock = Clock()
        with patch.object(cp.time, "monotonic", side_effect=lambda: clock.now), \
                patch.object(cp.time, "time", side_effect=[1000, 5000]):
            ws.handle_payload({"type": "executing", "data": {"node": "8"}})
            clock.tick(5)
            ws.handle_payload({"type": "executed", "data": {"node": "8"}})
        self.assertEqual(ws.node_ms["8"]["ms"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
