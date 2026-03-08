import requests

required_nodes = [
    "UNETLoader", "VAELoader", "CLIPVisionLoader", 
    "SVD_img2vid_Conditioning", "KSampler", "VAEDecode", 
    "CreateVideo", "SaveVideo", 
    "WanVideoModelLoader", "WanVideoVAELoader", 
    "LoadWanVideoT5TextEncoder", "WanVideoTextEncode", 
    "WanVideoEncode", "WanVideoImageToVideoEncode", 
    "WanVideoDecode"
]

try:
    response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
    if response.status_code == 200:
        available_nodes = response.json()
        print(f"Total nodes available: {len(available_nodes)}")
        
        print("\nChecking required nodes:")
        for node in required_nodes:
            if node in available_nodes:
                print(f"OK: {node} - Available")
                node_info = available_nodes[node]
                if 'input' in node_info:
                    inputs = node_info['input']
                    if 'required' in inputs:
                        print(f"  Required inputs: {list(inputs['required'].keys())}")
                    if 'optional' in inputs:
                        print(f"  Optional inputs: {list(inputs['optional'].keys())}")
                if 'output' in node_info:
                    print(f"  Outputs: {node_info['output']}")
            else:
                print(f"ERROR: {node} - NOT available")
    else:
        print(f"Error: Status code {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
