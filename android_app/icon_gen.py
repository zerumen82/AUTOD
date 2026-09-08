import os
import json
import socket
from PIL import Image, ImageDraw

def create_app_icon():
    img = Image.new("RGB", (512, 512), color=(2, 6, 23))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 100, 412, 412], fill=(6, 182, 212), outline=(168, 85, 247), width=12)
    icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
    img.save(icon_path)
    return icon_path

if __name__ == "__main__":
    icon = create_app_icon()
    print(f"Icono creado en: {icon}")
