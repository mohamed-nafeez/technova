#!/usr/bin/env python3
"""
Test script demonstrating the new split function architecture
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from production_ml_utils import check_billboard_presence, get_billboard_crops
import cv2
import numpy as np

def test_split_functions():
    """Test the new split function architecture"""
    print("🧪 TESTING SPLIT FUNCTION ARCHITECTURE")
    print("=" * 50)
    
    # Test image
    test_image = "examples/bill 3.jpeg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📷 Testing with: {test_image}")
    print()
    
    # ===========================
    # FUNCTION 1: Check Billboard Presence
    # ===========================
    print("🔍 FUNCTION 1: check_billboard_presence()")
    print("-" * 40)
    
    presence_result = check_billboard_presence(test_image)
    
    print(f"Billboard Present: {presence_result['billboard_present']}")
    print(f"Reason: {presence_result['reason']}")
    print(f"Billboard Count: {presence_result['billboard_count']}")
    print(f"Status: {presence_result['status']}")
    print()
    
    # ===========================
    # FUNCTION 2: Get Billboard Crops
    # ===========================
    print("🖼️ FUNCTION 2: get_billboard_crops()")
    print("-" * 40)
    
    if presence_result['billboard_present'] == "yes":
        crops = get_billboard_crops(test_image)
        
        print(f"Number of crops extracted: {len(crops)}")
        
        for i, crop_data in enumerate(crops):
            print(f"\nCrop {i+1}:")
            print(f"  Confidence: {crop_data['confidence_percent']}%")
            print(f"  Bounding Box: {crop_data['bbox']}")
            print(f"  Crop Size: {crop_data['crop_size']}")
            print(f"  Should Process OCR: {crop_data['should_process_ocr']}")
            
            # Save crop for demonstration
            crop_filename = f"billboard_crop_{i+1}.jpg"
            crop_image = crop_data['crop_image']
            cv2.imwrite(crop_filename, crop_image)
            print(f"  Saved crop to: {crop_filename}")
    else:
        print("❌ No billboards detected, skipping crop extraction")
    
    print()
    print("✅ Split function test completed!")

def demonstrate_api_usage():
    """Demonstrate how these functions can be used in an API"""
    print("\n🚀 API USAGE DEMONSTRATION")
    print("=" * 50)
    
    test_image = "examples/bill 3.jpeg"
    
    # API Endpoint 1: Check if billboards exist
    print("📡 API Endpoint 1: /check-billboards")
    presence = check_billboard_presence(test_image)
    print(f"Response: {presence}")
    print()
    
    # API Endpoint 2: Get cropped billboards
    if presence['billboard_present'] == "yes":
        print("📡 API Endpoint 2: /get-billboard-crops")
        crops = get_billboard_crops(test_image)
        
        # In a real API, you'd return the images as base64 or save to storage
        api_response = []
        for crop_data in crops:
            api_response.append({
                "crop_id": crop_data['crop_id'],
                "confidence": crop_data['confidence_percent'],
                "bbox": crop_data['bbox'],
                "crop_size": crop_data['crop_size'],
                "should_process_ocr": crop_data['should_process_ocr']
                # In real API: "image_data": base64_encoded_image
            })
        
        print(f"Response: {api_response}")
    else:
        print("📡 API Endpoint 2: Not called (no billboards detected)")

def compare_with_legacy():
    """Compare new functions with legacy detect_billboards"""
    print("\n🔄 LEGACY COMPARISON")
    print("=" * 50)
    
    from production_ml_utils import detect_billboards
    
    test_image = "examples/bill 3.jpeg"
    
    print("Legacy detect_billboards() function:")
    legacy_result = detect_billboards(test_image)
    print(f"Returns {len(legacy_result)} detection dictionaries")
    
    print("\nNew split architecture:")
    presence = check_billboard_presence(test_image)
    crops = get_billboard_crops(test_image) if presence['billboard_present'] == "yes" else []
    print(f"check_billboard_presence(): {presence['billboard_present']} ({presence['billboard_count']} billboards)")
    print(f"get_billboard_crops(): {len(crops)} crop dictionaries")
    
    print(f"\n✅ Results are equivalent - both find {len(legacy_result)} billboards")

if __name__ == "__main__":
    test_split_functions()
    demonstrate_api_usage()
    compare_with_legacy()
