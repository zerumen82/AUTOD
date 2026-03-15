import requests

def list_files_from_huggingface(repo_id, repo_type="model"):
    """List all files in a Hugging Face repository"""
    url = f"https://huggingface.co/api/{repo_type}s/{repo_id}/tree/main"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if response.status_code == 200:
            data = response.json()
            files = []
            
            for item in data:
                if "path" in item:
                    files.append(item["path"])
            
            return files
        else:
            print(f"Failed to list files. Status code: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    repo_id = "Kijai/WanVideo_comfy"
    print(f"Listing files in {repo_id}:")
    files = list_files_from_huggingface(repo_id)
    
    if files:
        # Filter for VAE files
        vae_files = [f for f in files if "vae" in f.lower() or f.endswith(".pth")]
        
        if vae_files:
            print(f"\nVAE-related files:")
            for vae_file in vae_files:
                print(f"- {vae_file}")
        
        print(f"\nAll files ({len(files)}):")
        for file in files:
            print(f"- {file}")
    else:
        print("No files found in the repository.")
