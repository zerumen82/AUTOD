import os

def find_svd_node():
    root_dir = "ui/tob/ComfyUI"
    search_pattern = "SVD_img2vid_Conditioning"
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if search_pattern in content:
                            print(f"Encontrado en: {file_path}")
                            # Buscar la definición de la clase
                            lines = content.split("\n")
                            for i, line in enumerate(lines):
                                if search_pattern in line:
                                    print(f"  Línea: {i+1}")
                                    # Mostrar 20 líneas antes y después
                                    start = max(0, i - 20)
                                    end = min(len(lines), i + 50)
                                    for j in range(start, end):
                                        print(f"  {j+1}: {lines[j]}")
                            return
                except:
                    continue
    
    print("No se encontró el nodo SVD_img2vid_Conditioning")

if __name__ == "__main__":
    find_svd_node()
