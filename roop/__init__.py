import os
import ctypes
import sys

# Try to add local project DLL directory to DLL search path as early as possible
try:
	dll_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dll'))
	if os.path.exists(dll_dir):
		try:
			os.add_dll_directory(dll_dir)
		except Exception:
			# os.add_dll_directory may fail on older Python versions; fallback to PATH append
			os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

		# Try to pre-load cuDNN DLL to avoid onnxruntime provider missing errors
		try:
			cudnn_path = os.path.join(dll_dir, 'cudnn64_9.dll')
			if os.path.exists(cudnn_path):
				ctypes.WinDLL(cudnn_path)
		except Exception:
			pass
except Exception:
	pass

from .types import Face, Frame