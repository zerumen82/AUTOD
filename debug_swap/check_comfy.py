# -*- coding: utf-8 -*-
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import requests

r = requests.get("http://127.0.0.1:8188/history", timeout=10)
if r.ok:
    data = r.json()
    print("History entries: {}".format(len(data)))
    for prompt_id, info in list(data.items())[-2:]:
        status = info.get("status", {})
        print("  Prompt {}... status={}".format(prompt_id[:8], status))
        if status.get("messages"):
            for msg in status["messages"][-3:]:
                print("    {}".format(msg))
        outputs = info.get("outputs", {})
        for node_id, node_out in outputs.items():
            images = node_out.get("images", [])
            for img in images:
                fn = img.get("filename", "?")
                print("    Node {}: image {}".format(node_id, fn))
else:
    print("History error:", r.status_code, r.text[:200])

r2 = requests.get("http://127.0.0.1:8188/queue", timeout=5)
if r2.ok:
    q = r2.json()
    print("Queue running: {}, pending: {}".format(
        len(q.get("queue_running", [])), len(q.get("queue_pending", []))))

# Check model files
r3 = requests.get("http://127.0.0.1:8188/object_info/UNETLoaderGGUF", timeout=10)
if r3.ok:
    info = r3.json()
    models = info.get("UNETLoaderGGUF", {}).get("input", {}).get("required", {}).get("unet_name", [])
    print("\nAvailable GGUF models: {}".format(len(models)))
    for m in models[:10]:
        print("  - {}".format(m))
    if "LongCat-Image-Edit-Q4_K_S.gguf" in models:
        print("  *** LongCat-Image-Edit-Q4_K_S.gguf FOUND ***")
    else:
        print("  *** LongCat-Image-Edit-Q4_K_S.gguf NOT FOUND ***")
else:
    print("GGUF models check error:", r3.status_code)
