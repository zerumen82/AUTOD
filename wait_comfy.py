import socket
import time

# Wait and check if port is listening
for i in range(30):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(('127.0.0.1', 8188))
        s.close()
        if result == 0:
            print(f"Port 8188 is LISTENING (checked at {i+1}s)")
            break
        else:
            print(f"Port 8188 not ready at {i+1}s")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(1)
