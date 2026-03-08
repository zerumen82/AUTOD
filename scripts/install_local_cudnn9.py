#!/usr/bin/env python3
"""scripts/install_local_cudnn9.py

Extrae un ZIP de cuDNN9 a `dll/cudnn9/` y verifica la carga de la DLL llamando a `cudnnGetVersion`.

Uso:
    python scripts/install_local_cudnn9.py --zip /ruta/a/cudnn-9.x.x-windows-x64.zip

Si no se pasa --zip, buscará en `third_party/cudnn9.zip`.
"""

import argparse
import os
import sys
import zipfile
import shutil
import ctypes
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Instala y verifica cuDNN9 localmente")
    p.add_argument("--zip", help="Ruta al ZIP de cuDNN9", default="third_party/cudnn9.zip")
    p.add_argument("--dest", help="Directorio destino (default: dll/cudnn9)", default="dll/cudnn9")
    return p.parse_args()


def extract_zip(zip_path: Path, dest: Path):
    print(f"Extrayendo '{zip_path}' -> '{dest}'")
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP no encontrado: {zip_path}")
    if dest.exists():
        print("Directorio destino ya existe — se eliminará para evitar mezclas (si quieres conservar, muévelo antes).")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)

    print("Extracción completada.")


def find_cudnn_DLLs(dest: Path):
    bin_dir_candidates = [dest / 'bin', dest, dest / 'windows' , dest / 'bin' / 'win64']
    dlls = []
    for cand in bin_dir_candidates:
        if cand.exists():
            for f in cand.glob('**/cudnn*.dll'):
                dlls.append(f)
    # fallback: search entire dest tree
    if not dlls:
        for f in dest.rglob('cudnn*.dll'):
            dlls.append(f)
    return dlls


def check_cudnn_version_from_dll(dll_path: Path):
    print(f"Intentando cargar DLL: {dll_path}")
    try:
        lib = ctypes.CDLL(str(dll_path))
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar la DLL: {e}")

    # Intentar llamar a cudnnGetVersion
    for restype in (ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int):
        try:
            func = getattr(lib, 'cudnnGetVersion')
            func.restype = restype
            ver = func()
            # version returned as integer like 90201
            return int(ver)
        except Exception:
            continue
    raise RuntimeError("La DLL se cargó pero no se pudo invocar cudnnGetVersion")


def pretty_version(v: int):
    # Formato esperado: MMmmpp => ej. 90201 -> 9.2.1
    major = v // 10000
    minor = (v % 10000) // 100
    patch = v % 100
    return f"{major}.{minor}.{patch} ({v})"


def main():
    args = parse_args()
    zip_path = Path(args.zip)
    dest = Path(args.dest)

    try:
        extract_zip(zip_path, dest)
    except Exception as e:
        print("ERROR: No se pudo extraer el ZIP:", e)
        sys.exit(2)

    dlls = find_cudnn_DLLs(dest)
    if not dlls:
        print("ERROR: No se encontraron DLLs 'cudnn*.dll' en la extracción. Revisa la estructura del ZIP.")
        print("Contenido extraído:")
        for p in sorted([str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()]):
            print('  -', p)
        sys.exit(3)

    print(f"Encontradas {len(dlls)} DLL(s) de cuDNN (mostrando las primeras 5):")
    for d in dlls[:5]:
        print('  -', d)

    # Preferir cudnn64_9.dll si existe
    selected = None
    for d in dlls:
        if 'cudnn64_9' in d.name.lower():
            selected = d
            break
    if not selected:
        selected = dlls[0]

    try:
        ver = check_cudnn_version_from_dll(selected)
    except Exception as e:
        print("ERROR: fallo al verificar la DLL:", e)
        sys.exit(4)

    print("Verificación correcta: cuDNN detectado ->", pretty_version(ver))

    # Crear enlace simbólico/copia a ruta fija dentro del repo para facilitar su uso
    repo_dest = Path('dll') / 'cudnn9'
    if repo_dest.exists():
        print(f"Nota: '{repo_dest}' ya existe — se actualizará limpiando la carpeta antes de copiar.")
        shutil.rmtree(repo_dest)
    shutil.copytree(dest, repo_dest)
    print(f"Copia de cuDNN9 a '{repo_dest}' completada.")

    # Mensaje final
    print('\nSiguiente(s) recomendación(es):')
    print(' - Si piensas compilar ONNX Runtime, apunta CMAKE -Dcudnn_root="'+ str(repo_dest.resolve()) + '"')

    # Intentar actualizar run.py para priorizar cudnn9 automáticamente (idempotente)
    def update_run_py(repo_run_py='run.py'):
        try:
            run_path = Path(repo_run_py)
            if not run_path.exists():
                print(f"Nota: {repo_run_py} no existe en el repo — omitiendo patch automático.")
                return
            content = run_path.read_text(encoding='utf-8')
            if 'cudnn9_dir' in content:
                print('run.py ya contiene lógica para cudnn9 — no se requieren cambios.')
                return
            # Reemplazar bloque antiguo que prioriza solo cudnn8 por la nueva lógica (cudnn9 > cudnn8)
            old_marker = "# 2. Priorizar carpeta `dll/cudnn8/` (si existe) para mantener compatibilidad con ONNX Runtime 1.23"
            if old_marker in content:
                new_block = '''# 2. Priorizar carpeta `dll/cudnn9/` (si existe) y fallback a `dll/cudnn8/` para compatibilidad
cudnn9_dir = os.path.join(os.path.dirname(__file__), 'dll', 'cudnn9')
cudnn8_dir = os.path.join(os.path.dirname(__file__), 'dll', 'cudnn8')

# Prioridad: cudnn9 > cudnn8
preferred_cudnn = None
for cand, name in ((cudnn9_dir, 'cuDNN9'), (cudnn8_dir, 'cuDNN8')):
    if os.path.exists(cand):
        preferred_cudnn = cand
        print(f"✓ Se detectó carpeta de {name}: {cand} — priorizando su carga")
        # Añadir antes al PATH y usar add_dll_directory
        if cand not in os.environ.get('PATH', ''):
            os.environ['PATH'] = cand + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(cand)
        except Exception:
            pass

        cudnn_files = sorted([f for f in os.listdir(cand) if f.lower().startswith('cudnn') and f.lower().endswith('.dll')])
        if cudnn_files:
            print(f"\n📦 Precargando {len(cudnn_files)} DLLs de cuDNN desde {cand}:")
            for fname in cudnn_files:
                fpath = os.path.join(cand, fname)
                try:
                    ctypes.CDLL(fpath)
                    print(f"  ✓ {fname}")
                except Exception as e:
                    print(f"  ✗ {fname}: {e}")
        # break after first (highest priority) candidate
        break'''
                content = content.replace(old_marker, new_block)
                backup = run_path.with_suffix('.py.bak')
                run_path.replace(backup)
                run_path.write_text(content, encoding='utf-8')
                print(f'run.py parcheado: {run_path} (backup: {backup})')
            else:
                print('run.py no contiene el marcador esperado; omitiendo parche automático.')
        except Exception as e:
            print('Fallo al intentar parchear run.py:', e)

    update_run_py()

    print('\nHecho.')


if __name__ == '__main__':
    main()
