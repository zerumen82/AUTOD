import requests

try:
    response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
    if response.status_code == 200:
        available_nodes = response.json()
        
        if "SVD_img2vid_Conditioning" in available_nodes:
            print("SVD_img2vid_Conditioning available")
            svd_node = available_nodes["SVD_img2vid_Conditioning"]
            
            if "input" in svd_node:
                inputs = svd_node["input"]
                if "required" in inputs:
                    print(f"Required inputs: {list(inputs['required'].keys())}")
                if "optional" in inputs:
                    print(f"Optional inputs: {list(inputs['optional'].keys())}")
            if "output" in svd_node:
                print(f"Outputs: {svd_node['output']}")
    else:
        print(f"Error: Status code {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
