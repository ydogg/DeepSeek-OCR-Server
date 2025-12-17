import requests
import base64
import sys
import json

def load_image_and_encode(image_path):
    """Load an image and encode it to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_ocr_endpoint(image_path, ocr_level="MD_MERGED"):
    """
    Test the original OCR endpoint with an image
    """
    # Read the image file and encode to base64
    image_data = load_image_and_encode(image_path)

    # Test original OCR endpoint
    url = "http://localhost:8001/v1/images/ocr"

    print(f"=== Testing Original OCR Endpoint (Level: {ocr_level}) ===")
    
    # Prepare the request payload
    payload = {
        "image": image_data,
        "level": ocr_level
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
            print("OCR RESULT:")
            print("=" * 50)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print()
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def test_health_endpoint():
    """
    Test the health check endpoint
    """
    url = "http://localhost:8001/health"
    
    print("=== Testing Health Check Endpoint ===")
    
    try:
        response = requests.get(url, timeout=30)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("HEALTH CHECK RESULT:")
            print("=" * 50)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print()
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def test_all_levels(image_path):
    """Test all OCR levels"""
    levels = ["RAW", "MD_IMAGE", "MD_TEXT", "MD_MERGED"]
    for level in levels:
        test_ocr_endpoint(image_path, level.lower())  # Convert to lowercase for API

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Use default test image if no argument provided
        image_path = "test_data/test_image.jpg"
    else:
        image_path = sys.argv[1]

    # Test health endpoint
    test_health_endpoint()

    # Test with MD_MERGED level by default
    test_ocr_endpoint(image_path, "md_merged")

    # Uncomment the following line to test all levels
    # test_all_levels(image_path)