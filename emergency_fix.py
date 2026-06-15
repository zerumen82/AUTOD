import sys
import os

file_path = 'roop/ProcessMgr.py'
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found")
    sys.exit(1)

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Localizar la función release_resources (la primera aparición legítima)
# El archivo debería terminar después de la definición de esa función.
new_lines = []
found_release = False
for line in lines:
    new_lines.append(line)
    if 'def release_resources(self):' in line:
        found_release = True
    elif found_release and 'self.Release()' in line:
        # Hemos encontrado el cuerpo de la función, el resto es basura duplicada
        break

with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
    f.writelines(new_lines)

print(f"Archivo reparado. Total líneas: {len(new_lines)}")
