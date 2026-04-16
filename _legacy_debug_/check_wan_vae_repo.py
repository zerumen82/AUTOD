import requests

repo_url = 'https://huggingface.co/api/models/Kijai/WanVideo_comfy'
try:
    response = requests.get(repo_url)
    response.raise_for_status()
    data = response.json()
    
    print('Checking available files in WanVideo_comfy repo...')
    print(f"Repo name: {data.get('id')}")
    print(f"Full name: {data.get('modelId')}")
    
    try:
        response = requests.get('https://huggingface.co/Kijai/WanVideo_comfy/raw/main/README.md')
        print('\nREADME contents:')
        print('-' * 50)
        print(response.text[:500])
    except:
        print('Could not get README')
        
except Exception as e:
    print(f'Error: {e}')
