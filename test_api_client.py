"""
Test client for Billboard Analysis FastAPI
Demonstrates how to use the API endpoints
"""

import requests
import json
import time
from pathlib import Path

class BillboardAPIClient:
    """Client for interacting with Billboard Analysis API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_image(self, image_path: str):
        """Analyze a single billboard image"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.base_url}/analyze",
                    files=files,
                    headers=self.headers
                )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_batch(self, image_paths: list):
        """Analyze multiple images in batch"""
        try:
            files = []
            for path in image_paths:
                files.append(('files', open(path, 'rb')))
            
            response = requests.post(
                f"{self.base_url}/analyze/batch",
                files=files,
                headers=self.headers
            )
            
            # Close files
            for _, file_obj in files:
                file_obj.close()
                
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_stats(self):
        """Get performance statistics"""
        try:
            response = requests.get(
                f"{self.base_url}/stats",
                headers=self.headers
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def validate_image(self, image_path: str):
        """Validate image file"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.base_url}/validate",
                    files=files,
                    headers=self.headers
                )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    """Test the Billboard Analysis API"""
    print("🧪 Billboard Analysis API Test Client")
    print("=" * 50)
    
    # Initialize client
    client = BillboardAPIClient()
    
    # Test 1: Health Check
    print("🏥 Testing health check...")
    health = client.health_check()
    print(f"Health Status: {health.get('status', 'unknown')}")
    
    if health.get('status') != 'healthy':
        print("❌ API not healthy, stopping tests")
        return
    
    # Test 2: Image Validation
    test_images = [
        "examples/img1.jpeg",
        "examples/img2.jpeg",
        "examples/bill 3.jpeg",
        "examples/sample billboard.jpeg"
    ]
    
    valid_images = []
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"✅ Validating: {img_path}")
            validation = client.validate_image(img_path)
            if validation.get('success'):
                valid_images.append(img_path)
                print(f"   Valid: {validation['data']['valid']}")
            else:
                print(f"   ❌ Validation failed: {validation.get('error')}")
        else:
            print(f"⚠️ Image not found: {img_path}")
    
    if not valid_images:
        print("❌ No valid images found for testing")
        return
    
    # Test 3: Single Image Analysis
    test_image = valid_images[0]
    print(f"\n📸 Testing single image analysis: {test_image}")
    start_time = time.time()
    result = client.analyze_image(test_image)
    analysis_time = time.time() - start_time
    
    if result.get('success'):
        data = result['data']
        print(f"✅ Analysis completed in {analysis_time:.2f}s")
        print(f"   Status: {data['status']}")
        print(f"   Billboards detected: {data['total_billboards']}")
        print(f"   Processing time: {data['processing_time']:.2f}s")
        print(f"   Recommendation: {data.get('recommendation', 'N/A')}")
        
        if data.get('billboard_results'):
            first_billboard = data['billboard_results'][0]
            print(f"   First billboard confidence: {first_billboard.get('detection_confidence', 0):.1f}%")
            extracted_text = first_billboard.get('extracted_text', '')[:50]
            print(f"   Extracted text: '{extracted_text}{'...' if len(extracted_text) == 50 else ''}'")
    else:
        print(f"❌ Analysis failed: {result.get('error')}")
    
    # Test 4: Batch Analysis
    if len(valid_images) > 1:
        print(f"\n📁 Testing batch analysis with {len(valid_images[:3])} images...")
        batch_images = valid_images[:3]  # Test with first 3 images
        start_time = time.time()
        batch_result = client.analyze_batch(batch_images)
        batch_time = time.time() - start_time
        
        if batch_result.get('success'):
            data = batch_result['data']
            print(f"✅ Batch analysis completed in {batch_time:.2f}s")
            print(f"   Total files: {data['total_files']}")
            print(f"   Successful analyses: {data['successful_analyses']}")
            print(f"   Total billboards: {data['total_billboards_detected']}")
            print(f"   Batch overall safe: {data['batch_overall_safe']}")
        else:
            print(f"❌ Batch analysis failed: {batch_result.get('error')}")
    
    # Test 5: Performance Stats
    print(f"\n📊 Testing performance stats...")
    stats = client.get_stats()
    if stats.get('success'):
        data = stats['data']
        print(f"✅ Stats retrieved")
        print(f"   Device: {data.get('device', 'unknown')}")
        print(f"   Optimization level: {data.get('optimization_level', 'unknown')}")
        print(f"   Expected performance: {data.get('expected_performance', 'unknown')}")
        runtime = data.get('runtime', {})
        print(f"   System status: {runtime.get('system_status', 'unknown')}")
    else:
        print(f"❌ Stats failed: {stats.get('error')}")
    
    print(f"\n🎉 API testing completed!")
    print(f"📖 Full API documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
