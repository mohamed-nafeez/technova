import requests
import os

def test_simple_api():
    """Simple test for the Billboard Detection API"""
    
    base_url = "http://127.0.0.1:8000"
    
    # Test 1: Health check
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health Status: {response.status_code}")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API Status: {health['status']}")
            print(f"ML Model Available: {health['ml_model_available']}")
        else:
            print(f"❌ Health check failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # Test 2: Billboard detection
    url = f"{base_url}/detect_billboard"
    image_path = "examples/bill 3.jpeg"
    
    if os.path.exists(image_path):
        print(f"\n📸 Testing API with: {image_path}")
        
        try:
            with open(image_path, "rb") as f:
                files = {"image": (image_path, f, "image/jpeg")}
                response = requests.post(url, files=files, timeout=60)
            
            print(f"📊 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success!")
                print(f"   Billboard Detected: {result['billboard_detected']}")
                print(f"   Reason: {result['reason']}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"❌ Image not found: {image_path}")

if __name__ == "__main__":
    test_simple_api()
