#!/usr/bin/env python3
"""
Comprehensive test script for DeepSeek OCR Server
Tests both OpenAI compatible and original OCR endpoints
"""

import requests
import base64
import sys
import json
import time

def load_image_and_encode(image_path):
    """Load an image and encode it to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_health_endpoint(base_url="http://localhost:8001"):
    """Test the health check endpoint"""
    url = f"{base_url}/health"
    
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
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def test_openai_endpoint(image_path, base_url="http://localhost:8001"):
    """Test the OpenAI compatible endpoint"""
    # Read the image file and encode to base64
    image_data = load_image_and_encode(image_path)

    url = f"{base_url}/v1/chat/completions"

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
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def test_ocr_endpoint(image_path, ocr_level="md_merged", base_url="http://localhost:8001"):
    """Test the original OCR endpoint"""
    # Read the image file and encode to base64
    image_data = load_image_and_encode(image_path)

    url = f"{base_url}/v1/ocr"

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
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def test_all_ocr_levels(image_path, base_url="http://localhost:8001"):
    """Test all OCR levels"""
    levels = ["raw", "md_image", "md_text", "md_merged"]
    results = []
    
    for level in levels:
        result = test_ocr_endpoint(image_path, level, base_url)
        results.append((level, result))
        time.sleep(1)  # Small delay between requests
    
    print("=== OCR Levels Test Summary ===")
    for level, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{level:12}: {status}")
    print()
    
    return all(result for _, result in results)

def main():
    if len(sys.argv) < 2:
        # Use default test image if no argument provided
        image_path = "test_data/test_image.jpg"
    else:
        image_path = sys.argv[1]
    
    base_url = "http://localhost:8001"
    
    print(f"Testing DeepSeek OCR Server at {base_url}")
    print(f"Using test image: {image_path}")
    print()
    
    # Test health endpoint
    health_ok = test_health_endpoint(base_url)
    
    # Test OpenAI compatible endpoint
    openai_ok = test_openai_endpoint(image_path, base_url)
    
    # Test original OCR endpoint with default level
    ocr_ok = test_ocr_endpoint(image_path, "md_merged", base_url)
    
    # Test all OCR levels
    all_levels_ok = test_all_ocr_levels(image_path, base_url)
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Health Check      : {'PASS' if health_ok else 'FAIL'}")
    print(f"OpenAI Endpoint   : {'PASS' if openai_ok else 'FAIL'}")
    print(f"OCR Endpoint      : {'PASS' if ocr_ok else 'FAIL'}")
    print(f"All OCR Levels    : {'PASS' if all_levels_ok else 'FAIL'}")
    print()
    
    overall_success = health_ok and openai_ok and ocr_ok and all_levels_ok
    print(f"Overall Result    : {'PASS' if overall_success else 'FAIL'}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())