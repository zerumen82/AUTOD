import os
import sys
import tempfile
import time

from PyQt5.QtWidgets import QApplication
import webview

publicurl = None
sdurl = None


def run_gradio_and_load_url():
    os.environ["QTWEBENGINE_REMOTE_DEBUGGING"] = "9222"
    app = QApplication(sys.argv)
    windowCreator()
    sys.exit(app.exec_())


def windowCreator():
    webview.create_window('AUTO-DEEP', publicurl)
    webview.start()


