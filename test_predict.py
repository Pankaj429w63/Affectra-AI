import urllib.request
import json

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = f'--{boundary}\r\nContent-Disposition: form-data; name="text"\r\n\r\nHello\r\n--{boundary}--\r\n'

req = urllib.request.Request(
    'https://affectra-ai.onrender.com/predict/raw',
    data=body.encode('utf-8'),
    method='POST',
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

try:
    with urllib.request.urlopen(req, timeout=30) as response:
        print('Status:', response.status)
        print('Response:', response.read().decode())
except urllib.error.HTTPError as e:
    print('HTTPError Status:', e.code)
    print('HTTPError Response:', e.read().decode())
except Exception as e:
    print('Error:', e)
