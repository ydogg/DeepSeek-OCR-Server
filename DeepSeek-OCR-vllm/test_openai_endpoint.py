import requests
import base64
import sys
import json

def test_openai_endpoint(image_path):
    """
    Test the OpenAI compatible endpoint with an image
    """
    # Read the image file and encode to base64
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Test OpenAI compatible endpoint
    url = "http://localhost:8001/v1/chat/completions"

    print("=== Testing OpenAI Compatible Endpoint ===")
    
    # Prepare the request payload
    payload = {
        "model": "deepseek-ai/DeepSeek-OCR",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("OpenAI COMPATIBLE RESULT:")
            print("=" * 50)
            print(result["choices"][0]["message"]["content"])
            print()
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Use default test image if no argument provided
        image_path = "test_data/test_image.jpg"
    else:
        image_path = sys.argv[1]

    test_openai_endpoint(image_path)