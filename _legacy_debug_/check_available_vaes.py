import requests

try:
    response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
    if response.status_code == 200:
        available_nodes = response.json()
        
        if "VAELoader" in available_nodes:
            print("VAELoader available")
            vae_node = available_nodes["VAELoader"]
            
            if "input" in vae_node and "required" in vae_node["input"]:
                vae_name_input = vae_node["input"]["required"]["vae_name"]
                print(f"Available VAEs: {vae_name_input[0]}")
            else:
                print("VAELoader doesn't have vae_name input")
    else:
        print(f"Error: Status code {response.status_code}")
except Exception as e:
    print(f"Error: {e}")

# Check what wanvideo expects for text encoders
try:
    response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
    if response.status_code == 200:
        available_nodes = response.json()
        
        if "LoadWanVideoT5TextEncoder" in available_nodes:
            print("\nLoadWanVideoT5TextEncoder available")
            t5_node = available_nodes["LoadWanVideoT5TextEncoder"]
            
            if "input" in t5_node and "required" in t5_node["input"]:
                model_name_input = t5_node["input"]["required"]["model_name"]
                print(f"Available text encoders: {model_name_input[0]}")
            else:
                print("LoadWanVideoT5TextEncoder doesn't have model_name input")
                
        if "WanVideoSampler" in available_nodes:
            print("\nWanVideoSampler available")
            sampler_node = available_nodes["WanVideoSampler"]
            
            if "input" in sampler_node and "required" in sampler_node["input"]:
                scheduler_input = sampler_node["input"]["required"]["scheduler"]
                print(f"Available schedulers: {scheduler_input[0]}")
            else:
                print("WanVideoSampler doesn't have scheduler input")
    else:
        print(f"Error: Status code {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
