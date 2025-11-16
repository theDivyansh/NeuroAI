#!/usr/bin/env python3
import requests
from PIL import Image
import json

# Create test image
img = Image.new('RGB', (224, 224), color=(100, 100, 100))
img.save('test_image.png')

# Send to backend
print("Testing /api/predict endpoint...")
with open('test_image.png', 'rb') as f:
    try:
        r = requests.post('http://localhost:5000/api/predict', files={'image': f})
        print(f'Status: {r.status_code}')
        print(f'Response:')
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print(f'Error: {e}')
