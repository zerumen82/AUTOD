import requests
import time

try:
    # Get queue status
    response = requests.get("http://127.0.0.1:8188/queue", timeout=5)
    if response.status_code == 200:
        queue_status = response.json()
        print("Queue Status:")
        print(f"  Running: {queue_status['status']['exec_info']['running']}")
        print(f"  Pending: {queue_status['status']['exec_info']['pending']}")
        print(f"  Failed: {queue_status['status']['exec_info']['failed']}")
        
        # Get prompt history
        response = requests.get("http://127.0.0.1:8188/history", timeout=5)
        if response.status_code == 200:
            history = response.json()
            print("\nHistory:")
            if history:
                for prompt_id, data in history.items():
                    print(f"  Prompt ID: {prompt_id}")
                    print(f"  Status: {data.get('status', 'unknown')}")
                    print(f"  Node Errors: {data.get('node_errors', {})}")
                    print(f"  Total time: {data.get('total_time', 0)} seconds")
                    print(f"  Outputs: {list(data.get('outputs', {}).keys())}")
                    print()
            else:
                print("  No prompts processed yet")
    else:
        print(f"Error: Status code {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
