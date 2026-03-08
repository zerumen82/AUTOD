import os
import sys
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DLL_SRC = os.path.join(PROJECT_ROOT, 'dll', 'cudnn64_9.dll')
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"


def check_cudnn_dll():
    found_paths = []
    if os.path.exists(DLL_SRC):
        found_paths.append(('project_dll', DLL_SRC))

    cppath = os.path.join(CUDA_BIN, 'cudnn64_9.dll')
    if os.path.exists(cppath):
        found_paths.append(('cuda_bin', cppath))

    # Try also to search in PATH
    for p in os.environ.get('PATH', '').split(os.pathsep):
        p = p.strip('"')
        if not p:
            continue
        candidate = os.path.join(p, 'cudnn64_9.dll')
        if os.path.exists(candidate):
            found_paths.append((p, candidate))

    return found_paths


def check_onnxruntime_providers():
    try:
        import onnxruntime as ort
        prov = ort.get_available_providers()
        return prov, ort.__version__
    except Exception as e:
        return None, str(e)


def copy_to_cuda_bin(src):
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    if not os.path.exists(CUDA_BIN):
        raise FileNotFoundError(CUDA_BIN)

    dst = os.path.join(CUDA_BIN, os.path.basename(src))
    # Ask for confirmation
    print(f"About to copy {src} -> {dst}. This requires admin rights and may overwrite existing files.")
    confirm = input('Proceed? (yes/no): ').strip().lower()
    if confirm not in ('yes', 'y'):
        print('Aborting.')
        return False

    try:
        shutil.copy2(src, dst)
        print(f'Copied {src} -> {dst}')
        return True
    except Exception as e:
        print(f'Copy failed: {e}')
        return False


if __name__ == '__main__':
    print('Checking cudnn dll presence (project + cuda bin + PATH):')
    found = check_cudnn_dll()
    if found:
        for location, path in found:
            print(f'  - {location}: {path}')
    else:
        print('  - cudnn64_9.dll NOT found in project, CUDA bin, or PATH')

    print('\nChecking onnxruntime providers:')
    prov, version = check_onnxruntime_providers()
    if prov is None:
        print('  onnxruntime not available or error:', version)
    else:
        print(f'  onnxruntime version: {version}\n  providers: {prov}')

    if not any(p[0] == 'cuda_bin' for p in found):
        # Offer to copy if project dll exists
        if any(p[0] == 'project_dll' for p in found):
            src = next(p[1] for p in found if p[0] == 'project_dll')
            print('\nDo you want to copy the cudnn DLL to detected CUDA bin location?')
            try:
                copy_to_cuda_bin(src)
            except Exception as e:
                print('Error while copying:', e)
        else:
            print('\nNo local cudnn64_9.dll found in project dir to copy. Download cuDNN from NVIDIA and place the DLL in project/dll or in your CUDA bin.')

    print('\nDone.')
