#!/usr/bin/env python3
"""
Test script for Billboard API - all three endpoints
"""
import requests
import json

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_api_endpoints():
    print("🚀 Testing Billboard API Endpoints")
    print("=" * 50)
    
    # Test image file
    test_image = "examples/bill 3.jpeg"
    
    try:
        # 1. Test billboard detection
        print("\n1️⃣ Testing Billboard Detection (/detect_billboard)")
        with open(test_image, 'rb') as f:
            response = requests.post(
                f"{BASE_URL}/detect_billboard",
                files={"image": f}
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Detection Result: {result}")
        else:
            print(f"❌ Detection failed: {response.status_code} - {response.text}")
            return
        
        # 2. Test billboard cropping (only if detection was successful)
        if result.get("billboard_detected"):
            print("\n2️⃣ Testing Billboard Cropping (/crop_billboard)")
            with open(test_image, 'rb') as f:
                response = requests.post(
                    f"{BASE_URL}/crop_billboard",
                    files={"image": f}
                )
            
            if response.status_code == 200:
                print("✅ Cropping successful - saving cropped image")
                with open("cropped_billboard.jpg", "wb") as f:
                    f.write(response.content)
                print("📁 Saved as 'cropped_billboard.jpg'")
                
                # 3. Test OCR on cropped image
                print("\n3️⃣ Testing OCR Text Extraction (/extract_text)")
                with open("cropped_billboard.jpg", 'rb') as f:
                    response = requests.post(
                        f"{BASE_URL}/extract_text",
                        files={"image": f}
                    )
                
                if response.status_code == 200:
                    ocr_result = response.json()
                    print(f"✅ OCR Result:")
                    print(f"   📝 Text: '{ocr_result.get('text', '')}'")
                    print(f"   🎯 Confidence: {ocr_result.get('confidence', 0):.2f}")
                    print(f"   📊 Word Count: {ocr_result.get('word_count', 0)}")
                    print(f"   💬 Message: {ocr_result.get('message', '')}")
                else:
                    print(f"❌ OCR failed: {response.status_code} - {response.text}")
            else:
                print(f"❌ Cropping failed: {response.status_code} - {response.text}")
        else:
            print("⚠️ No billboard detected, skipping cropping and OCR tests")
    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running:")
        print("   uvicorn billboard_api:app --reload")
    except FileNotFoundError:
        print(f"❌ Test image not found: {test_image}")
        print("   Please make sure the test image exists")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_health_check():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"🏥 Health Check: {health}")
            return health.get("ml_model_available", False)
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🏥 Checking API Health...")
    if test_health_check():
        test_api_endpoints()
    else:
        print("❌ API is not healthy or ML model not available")
    
    print("\n🎉 Testing completed!")
