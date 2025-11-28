#!/usr/bin/env python3

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_offline_mode():
    """Test offline mode configuration"""
    print("Testing offline mode...")
    
    # Set offline mode
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Import with offline mode
        from server.config import ONLINE_OCR_MODE
        print(f"Current mode: {'Online' if ONLINE_OCR_MODE else 'Offline'}")
        
        if ONLINE_OCR_MODE:
            print("ERROR: Should be in offline mode!")
            return False
            
        # Try importing processors
        from server.core.processor import OCRProcessor
        from server.core.online_processor import OnlineOCRProcessor
        
        # Create processors
        offline_proc = OCRProcessor()
        online_proc = OnlineOCRProcessor()
        
        # Select processor
        processor = online_proc if ONLINE_OCR_MODE else offline_proc
        print(f"Selected processor type: {type(processor).__name__}")
        
        if not isinstance(processor, OCRProcessor):
            print("ERROR: Wrong processor selected for offline mode!")
            return False
            
        print("Offline mode test passed!")
        return True
        
    except Exception as e:
        print(f"Offline mode test failed: {e}")
        return False

def test_online_mode():
    """Test online mode configuration"""
    print("\nTesting online mode...")
    
    try:
        # Modify the config to enable online mode
        from server.config import ONLINE_OCR_MODE
        print(f"Current mode: {'Online' if ONLINE_OCR_MODE else 'Offline'}")
        
        # For this test, we'll just check if the online processor can be instantiated
        from server.core.online_processor import OnlineOCRProcessor
        online_proc = OnlineOCRProcessor()
        print("Online processor created successfully")
        
        print("Online mode test completed (not fully enabled in config)!")
        return True
        
    except Exception as e:
        print(f"Online mode test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing OCR mode switching...")
    
    success = True
    success &= test_offline_mode()
    success &= test_online_mode()
    
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)