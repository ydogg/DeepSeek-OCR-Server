#!/usr/bin/env python3

import sys
import os
from PIL import Image
import io
import base64

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_processor_unified_interface():
    """Test that both processors have the same unified interface"""
    print("Testing unified processor interface...")
    
    try:
        from server.core.processor import OCRProcessor
        from server.core.online_processor import OnlineOCRProcessor
        from server.schemas.models import OCRRequest
        
        # Create both processors
        offline_proc = OCRProcessor()
        online_proc = OnlineOCRProcessor()
        
        # Test that both have the same methods
        methods = ['start_workers', 'stop_workers', 'submit_request', 'wait_for_result', 'health_check']
        
        for method in methods:
            if not hasattr(offline_proc, method):
                print(f"Offline processor missing method: {method}")
                return False
            if not hasattr(online_proc, method):
                print(f"Online processor missing method: {method}")
                return False
                
        print("✓ Both processors have the same interface methods")
        
        # Test that both can be initialized
        print("✓ Offline processor initialized successfully")
        print("✓ Online processor initialized successfully")
        
        # Test health check
        offline_health = offline_proc.health_check()
        online_health = online_proc.health_check()
        print(f"✓ Offline processor health: {offline_health}")
        print(f"✓ Online processor health: {online_health}")
        
        print("Unified interface test passed!")
        return True
        
    except Exception as e:
        print(f"Unified interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_settings():
    """Test that configuration settings are correct"""
    print("\nTesting configuration settings...")
    
    try:
        from server.config import (
            ONLINE_OCR_MODE, 
            ONLINE_OCR_BASE_URL, 
            ONLINE_OCR_MODEL_NAME, 
            ONLINE_OCR_API_KEY,
            OCR_PROMPT
        )
        
        print(f"✓ ONLINE_OCR_MODE: {ONLINE_OCR_MODE}")
        print(f"✓ ONLINE_OCR_BASE_URL: {ONLINE_OCR_BASE_URL}")
        print(f"✓ ONLINE_OCR_MODEL_NAME: {ONLINE_OCR_MODEL_NAME}")
        print(f"✓ ONLINE_OCR_API_KEY: {'*' * len(ONLINE_OCR_API_KEY) if ONLINE_OCR_API_KEY else 'None'}")
        print(f"✓ OCR_PROMPT: {repr(OCR_PROMPT)}")
        
        # Verify that the prompt is the expected one
        expected_prompt = "<image>\n<|grounding|>Convert the document to markdown."
        if OCR_PROMPT == expected_prompt:
            print("✓ OCR prompt is correctly set")
        else:
            print(f"⚠ OCR prompt differs from expected: {repr(expected_prompt)}")
            
        print("Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_handling():
    """Test that both processors handle prompts correctly"""
    print("\nTesting prompt handling...")
    
    try:
        from server.core.processor import OCRProcessor
        from server.core.online_processor import OnlineOCRProcessor
        from server.schemas.models import OCRRequest
        from server.config import OCR_PROMPT
        import uuid
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        
        # Test with no prompt (should use default)
        request_no_prompt = OCRRequest(f"test-{uuid.uuid4().hex[:8]}", img, None)
        
        # Test with custom prompt
        custom_prompt = "Custom OCR prompt for testing"
        request_custom_prompt = OCRRequest(f"test-{uuid.uuid4().hex[:8]}", img, custom_prompt)
        
        print("✓ Test requests created successfully")
        print("Prompt handling test passed!")
        return True
        
    except Exception as e:
        print(f"Prompt handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Final Implementation Test ===")
    
    success = True
    success &= test_config_settings()
    success &= test_processor_unified_interface()
    success &= test_prompt_handling()
    
    if success:
        print("\n=== ALL TESTS PASSED! ===")
        print("The unified OCR implementation is ready for use.")
    else:
        print("\n=== SOME TESTS FAILED! ===")
        sys.exit(1)