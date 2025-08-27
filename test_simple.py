"""
Test client for simple API
"""
import requests

def test_simple_api():
    base_url = "http://127.0.0.1:8001"
    
    print("🏥 Testing simple API health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Cannot connect to simple API: {e}")
        return
    
    print("\n📤 Testing file upload...")
    try:
        with open("examples/bill 3.jpeg", "rb") as f:
            files = {"image": ("bill 3.jpeg", f, "image/jpeg")}
            response = requests.post(f"{base_url}/test_upload", files=files, timeout=30)
            
        print(f"📊 Upload test: {response.status_code}")
        print(f"🔍 Response: {response.json()}")
        
    except Exception as e:
        print(f"❌ Upload test failed: {e}")

if __name__ == "__main__":
    test_simple_api()
