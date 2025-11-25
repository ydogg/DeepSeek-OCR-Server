import requests
import json

# Test data - using the standard format you provided
test_request = {
    "model": "deepseek-ocr",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                    }
                },
                {
                    "type": "text",
                    "text": "Free OCR."
                }
            ]
        }
    ]
}

# Send request
try:
    response = requests.post("http://localhost:8000/v1/chat/completions", json=test_request, timeout=30)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")