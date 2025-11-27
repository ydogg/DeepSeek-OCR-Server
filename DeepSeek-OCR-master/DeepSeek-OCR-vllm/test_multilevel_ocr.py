import requests
import json
import base64
import sys

def test_multilevel_ocr(image_path):
    """
    Test the multi-level OCR results feature
    """
    # Read the image file and encode to base64
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Test all three levels
    url = "http://localhost:8000/v1/ocr"

    # Test 1: Raw level
    print("=== Testing RAW Level ===")
    payload_raw = {
        "image": image_data,
        "level": "raw",
        "save_results": True
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload_raw,
            timeout=120
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("1. RAW RESULT (with ref/det tags):")
            print("=" * 50)
            print(result["result"])
            print("\n")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

    # Test 2: Clean level
    print("=== Testing CLEAN Level ===")
    payload_clean = {
        "image": image_data,
        "level": "clean",
        "save_results": True
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload_clean,
            timeout=120
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("2. CLEANED RESULT:")
            print("=" * 50)
            print(result["result"])
            print("\n")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

    # Test 3: Image clean level
    print("=== Testing IMAGE_CLEAN Level ===")
    payload_image_clean = {
        "image": image_data,
        "level": "image_clean"
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload_image_clean,
            timeout=120
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("3. VL ANALYZED RESULT:")
            print("=" * 50)
            print(result["result"])
            print("\n")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_multilevel_ocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    test_multilevel_ocr(image_path)