"""
Example usage of the Billboard Analysis System
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from production_ml_utils import ProductionBillboardAnalyzer
import json
import time

def main():
    print("🎯 Billboard Analysis System - Example Usage")
    print("=" * 50)
    
    # Initialize the analyzer
    print("📱 Initializing analyzer...")
    analyzer = ProductionBillboardAnalyzer()
    
    # Health check
    print("🔍 Running health check...")
    health = analyzer.health_check()
    print(f"Status: {health['status']}")
    print(f"Models loaded: {health['models_loaded']}")
    
    # ⭐ CHANGE THESE IMAGE PATHS TO YOUR IMAGES ⭐
    example_images = [
        "examples/img1.jpeg",           # Change to your image path
        "examples/img2.jpeg",           # Change to your image path  
        "examples/sample billboard.jpeg", # Change to your image path
        "examples/bill 3.jpeg"          # Change to your image path
    ]
    
    for image_path in example_images:
        if os.path.exists(image_path):
            print(f"\n📸 Analyzing: {image_path}")
            start_time = time.time()
            
            try:
                result = analyzer.analyze_image(image_path)
                processing_time = time.time() - start_time
                
                print(f"⚡ Processing time: {processing_time:.2f}s")
                print(f"📊 Billboards detected: {result.get('total_billboards', 0)}")
                print(f"📝 Status: {result.get('status', 'unknown')}")
                
                if result.get('billboard_results'):
                    first_billboard = result['billboard_results'][0]
                    print(f"🎯 First billboard confidence: {first_billboard.get('detection_confidence', 0):.2f}%")
                    if first_billboard.get('extracted_text'):
                        print(f"📖 Extracted text: {first_billboard['extracted_text'][:100]}...")
                        
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                
            break  # Only test first available image
        else:
            print(f"⚠️ Image not found: {image_path}")
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    main()
