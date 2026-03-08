import requests

try:
    # Obtener info de nodos para ver las opciones de VAELoader
    response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
    if response.status_code == 200:
        object_info = response.json()
        if "VAELoader" in object_info:
            vae_node_info = object_info["VAELoader"]
            if "input" in vae_node_info and "required" in vae_node_info["input"] and "vae_name" in vae_node_info["input"]["required"]:
                vae_options = vae_node_info["input"]["required"]["vae_name"]
                print("Opciones disponibles para VAELoader:")
                print(vae_options)
            else:
                print("No se encontraron opciones para vae_name")
        else:
            print("VAELoader no está disponible")
    else:
        print(f"Error: Status code {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
