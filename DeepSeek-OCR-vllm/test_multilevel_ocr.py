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

    # Add save_results for md_merged level
    if level == "md_merged":
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

    # Test all levels
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

    # Test 2: md_image level
    print("=== Testing MD_IMAGE Level ===")
    payload_md_image = {
        "image": image_data,
        "level": "md_image",
        "save_results": True
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload_md_image,
            timeout=120
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("2. MARKDOWN IMAGE RESULT (with image tags preserved):")
            print("=" * 50)
            print(result["result"])
            print()
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

    # Test 3: md_text level
    print("=== Testing MD_TEXT Level ===")
    payload_md_text = {
        "image": image_data,
        "level": "md_text",
        "save_results": True
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload_md_text,
            timeout=120
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("3. MARKDOWN TEXT RESULT (all tags removed):")
            print("=" * 50)
            print(result["result"])
            print()
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

    # Test 4: md_merged level
    print("=== Testing MD_MERGED Level ===")
    payload_md_merged = {
        "image": image_data,
        "level": "md_merged"
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload_md_merged,
            timeout=120
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("4. MARKDOWN MERGED RESULT:")
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
        print("  level: 'raw', 'md_image', 'md_text', 'md_merged', or 'all' (default)")
        sys.exit(1)

    image_path = sys.argv[1]
    level = sys.argv[2] if len(sys.argv) > 2 else "all"

    if level == "all":
        test_multilevel_ocr(image_path)
    elif level in ["raw", "md_image", "md_text", "md_merged"]:
        test_ocr_level(image_path, level)
    else:
        print("Invalid level. Use 'raw', 'md_image', 'md_text', 'md_merged', or 'all'")
        sys.exit(1)
