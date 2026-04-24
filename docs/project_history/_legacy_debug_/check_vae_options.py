
import requests
import json

try:
    response = requests.get('http://127.0.0.1:8188/object_info', timeout=5)
    if response.status_code == 200:
        object_info = response.json()
        
        # Check VAELoader node
        if 'VAELoader' in object_info:
            vae_loader = object_info['VAELoader']
            if 'input' in vae_loader and 'required' in vae_loader['input'] and 'vae_name' in vae_loader['input']['required']:
                vae_options = vae_loader['input']['required']['vae_name'][0]
                print("Available VAE options:")
                for option in vae_options:
                    print(f"  - {option}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    print(traceback.format_exc())
