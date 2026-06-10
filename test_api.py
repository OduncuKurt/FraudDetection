import urllib.request
import json

req = urllib.request.Request('http://127.0.0.1:8000/api/trigger_new_fraud', method='POST')
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Error: {e}")
