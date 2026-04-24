
import requests
import json

try:
    response = requests.get('http://127.0.0.1:8188/object_info', timeout=5)
    if response.status_code == 200:
        object_info = response.json()
        print('Number of available nodes:', len(object_info))
        
        # Check for specific Wan and GGUF nodes
        wan_nodes = [name for name in object_info if 'Wan' in name or 'GGUF' in name]
        print('Wan/GGUF nodes available:', len(wan_nodes))
        
        if wan_nodes:
            for node_name in sorted(wan_nodes):
                print(f'  - {node_name}')
        else:
            print('No Wan or GGUF nodes found!')
            
        print()
        
        # Check for Roop and LTX nodes
        print('Roop nodes available:', [name for name in object_info if 'roop' in name.lower()])
        print('LTX nodes available:', [name for name in object_info if 'LTX' in name])
        print('CogVideo nodes available:', [name for name in object_info if 'CogVideo' in name])
        
    else:
        print('Error:', response.status_code, response.text)
except Exception as e:
    print('Request failed:', e)
