import requests

try:
    response = requests.get("http://127.0.0.1:8188/queue", timeout=5)
    if response.status_code == 200:
        print("Queue response:")
        print(response.text)
    else:
        print(f"Error: Status code {response.status_code}")
except Exception as e:
    print(f"Error: {e}")

try:
    response = requests.get("http://127.0.0.1:8188/history", timeout=5)
    if response.status_code == 200:
        print("\nHistory response:")
        print(response.text)
    else:
        print(f"Error: Status code {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
