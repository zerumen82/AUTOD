import requests

try:
    response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
    if response.status_code == 200:
        available_nodes = response.json()
        
        if "CLIPVisionLoader" in available_nodes:
            print("CLIPVisionLoader available")
            clip_vision_node = available_nodes["CLIPVisionLoader"]
            
            if "input" in clip_vision_node and "required" in clip_vision_node["input"]:
                clip_name_input = clip_vision_node["input"]["required"]["clip_name"]
                print(f"Available CLIP Vision models: {clip_name_input[0]}")
            else:
                print("CLIPVisionLoader doesn't have clip_name input")
    else:
        print(f"Error: Status code {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
