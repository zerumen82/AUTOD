
import webview
import sys

def open_sd():
    window = webview.create_window(
        "Stable Diffusion",
        "http://127.0.0.1:9871",
        width=1400,
        height=900,
        resizable=True,
        background_color='#1a1a1a'
    )
    webview.start(debug=False)

if __name__ == "__main__":
    open_sd()
