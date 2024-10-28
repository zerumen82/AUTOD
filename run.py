import os
import sys
import ctypes
import threadpoolctl
import warnings

# Suprimir advertencias específicas de RuntimeWarning
warnings.filterwarnings("ignore", message="Found Intel OpenMP.*")

# Determinar la ruta base
base_path = os.path.abspath(".")

# Ruta completa al archivo DLL
dll_path = os.path.join(base_path, '.venv', 'Lib', 'site-packages', 'torch', 'lib', 'libiomp5md.dll')

# Descargar LLVM OpenMP configurando variables de entorno para priorizar Intel OpenMP
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

# Cargar Intel OpenMP primero
try:
    ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
except OSError as e:
    warnings.warn(f"Failed to load Intel OpenMP: {e}", RuntimeWarning)

# Limitar las bibliotecas OpenMP para evitar conflictos
threadpoolctl.threadpool_limits(limits=1, user_api='openmp')

from roop import core

def main():
    core.run()

if __name__ == '__main__':
    main()