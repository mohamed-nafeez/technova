import requests
import os

def test_billboard_detection(image_path):
    """Test the billboard detection endpoint"""
    print(f"Testing billboard detection with: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post('http://127.0.0.1:8000/detect_billboard', files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

def test_billboard_cropping(image_path, output_path):
    """Test the billboard cropping endpoint"""
    print(f"\nTesting billboard cropping with: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post('http://127.0.0.1:8000/crop_billboard', files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        # Save the cropped image
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Cropped billboard saved to: {output_path}")
        return True
    else:
        try:
            print(f"❌ Error: {response.json()}")
        except:
            print(f"❌ Error: {response.text}")
        return False

def test_health_check():
    """Test the health endpoint"""
    try:
        response = requests.get('http://127.0.0.1:8000/health')
        print(f"Health Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def main():
    """Test both API endpoints"""
    print("🧪 Testing Billboard Detection & Cropping API")
    print("=" * 60)
    
    # First test health
    if not test_health_check():
        print("❌ API server not healthy")
        return
    
    # Test with sample images
    test_images = [
        "examples/img1.jpeg",
        "examples/img2.jpeg", 
        "examples/bill 3.jpeg",
        "examples/sample billboard.jpeg"
    ]
    
    for img in test_images:
        if os.path.exists(img):
            print(f"\n{'='*60}")
            print(f"🖼️ Testing with: {img}")
            print(f"{'='*60}")
            
            # Test detection first
            detection_result = test_billboard_detection(img)
            
            # If exactly one billboard detected, test cropping
            if detection_result.get('billboard_detected'):
                output_file = f"cropped_{os.path.basename(img)}"
                success = test_billboard_cropping(img, output_file)
                if success:
                    print(f"🎉 Successfully cropped and saved billboard!")
            else:
                print("⚠️ Skipping cropping - not exactly one billboard detected")
            
            break  # Test with first available image
        else:
            print(f"⚠️ Image not found: {img}")
    
    print(f"\n✅ Testing completed!")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("💡 Make sure the FastAPI server is running with:")
        print("   python -m uvicorn billboard_api:app --reload")
