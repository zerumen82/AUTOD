import requests
try:
    response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
    data = response.json()
    # Buscar nodos que contengan "Wan" y "Sampler"
    wan_sampler_nodes = [node for node in data.keys() if "Wan" in node and "Sampler" in node]
    print("Wan sampler nodes:", wan_sampler_nodes)
    
    if wan_sampler_nodes:
        node_name = wan_sampler_nodes[0]
        print(f"\nDetails for {node_name}:")
        if 'input' in data[node_name]:
            inputs = data[node_name]['input']
            if 'required' in inputs:
                print(f"  Required inputs: {list(inputs['required'].keys())}")
            if 'optional' in inputs:
                print(f"  Optional inputs: {list(inputs['optional'].keys())}")
        if 'output' in data[node_name]:
            print(f"  Outputs: {data[node_name]['output']}")
except Exception as e:
    print(f"Error: {e}")
