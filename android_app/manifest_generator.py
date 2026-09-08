import os
import json
import socket
import gradio as gr

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def generate_manifest():
    ip = get_local_ip()
    manifest = {
        "short_name": "AutoAuto Edit",
        "name": "AutoAuto Image Editor AI",
        "icons": [
            {
                "src": "https://gradio.s3-us-west-2.amazonaws.com/static/favicon.ico",
                "sizes": "192x192 512x512",
                "type": "image/x-icon"
            }
        ],
        "start_url": "/",
        "background_color": "#020617",
        "theme_color": "#020617",
        "display": "standalone",
        "orientation": "portrait"
    }
    return manifest

if __name__ == "__main__":
    ip = get_local_ip()
    print(f"IP local detectada para Android APK: {ip}")
