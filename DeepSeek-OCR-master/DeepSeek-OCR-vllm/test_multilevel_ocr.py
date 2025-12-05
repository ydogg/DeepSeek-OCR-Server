import requests
import json
import base64
import sys

def test_ocr_level(image_path, level):
    """
    Test OCR for a specific level
    """
    # Read the image file and encode to base64
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Test specific level
    url = "http://localhost:8001/v1/ocr"

    print(f"=== Testing {level.upper()} Level ===")
    payload = {
        "image": image_data,
        "level": level
    }

    # Add save_results for image_clean level
    if level == "image_clean":
        payload["save_results"] = True

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
            print(f"{level.upper()} RESULT:")
            print("=" * 50)
            print(result["result"])
            print()
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def test_multilevel_ocr(image_path):
    """
    Test the multi-level OCR results feature - all levels
    """
    # Read the image file and encode to base64
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Test all three levels
    url = "http://localhost:8001/v1/ocr"

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
            print()
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
            print()
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
            print()
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_multilevel_ocr.py <image_path> [level]")
        print("  level: 'raw', 'clean', 'image_clean', or 'all' (default)")
        sys.exit(1)

    image_path = sys.argv[1]
    level = sys.argv[2] if len(sys.argv) > 2 else "all"

    if level == "all":
        test_multilevel_ocr(image_path)
    elif level in ["raw", "clean", "image_clean"]:
        test_ocr_level(image_path, level)
    else:
        print("Invalid level. Use 'raw', 'clean', 'image_clean', or 'all'")
        sys.exit(1)
