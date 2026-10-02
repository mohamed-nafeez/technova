"""
Quick test script - Change image path here
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from production_ml_utils import ProductionBillboardAnalyzer

# ⭐ CHANGE THIS TO YOUR IMAGE PATH ⭐
IMAGE_PATH = "examples/img1.jpeg"  # <-- Change this path

def quick_test():
    print("🚀 Quick Billboard Analysis Test")
    
    analyzer = ProductionBillboardAnalyzer()
    
    if os.path.exists(IMAGE_PATH):
        print(f"📸 Testing with: {IMAGE_PATH}")
        result = analyzer.analyze_image(IMAGE_PATH)
        
        print(f"✅ Status: {result.get('status')}")
        print(f"📊 Billboards found: {result.get('total_billboards', 0)}")
        print(f"⏱️ Processing time: {result.get('processing_time', 0):.2f}s")
        
        return result
    else:
        print(f"❌ Image not found: {IMAGE_PATH}")
        print("💡 Update IMAGE_PATH variable to point to your image")
        return None

if __name__ == "__main__":
    quick_test()
