"""
Example usage of the Billboard Analysis System
"""

from src.production_ml_utils import BillboardAnalyzer
import json
import time

def main():
    print("🎯 Billboard Analysis System - Example Usage")
    print("=" * 50)
    
    # Initialize the analyzer
    print("📱 Initializing analyzer...")
    analyzer = BillboardAnalyzer()
    
    # Health check
    print("🔍 Running health check...")
    health = analyzer.health_check()
    print(f"Status: {health['status']}")
    print(f"Models loaded: {health['models_loaded']}")
    
    # Example image analysis
    example_images = [
        "examples/img1.jpeg",
        "examples/img2.jpeg", 
        "examples/sample billboard.jpeg"
    ]
    
    for image_path in example_images:
        print(f"\n📸 Analyzing: {image_path}")
        start_time = time.time()
        
        try:
            result = analyzer.analyze_image(image_path)
            processing_time = time.time() - start_time
            
            print(f"⚡ Processing time: {processing_time:.2f}s")
            print(f"📊 Billboards detected: {result.get('billboards_detected', 0)}")
            print(f"📝 Text blocks found: {result.get('total_text_blocks', 0)}")
            print(f"🛡️ Vulnerability score: {result.get('vulnerability_score', 0):.4f}")
            
            # Show first billboard result
            if result.get('results'):
                first_billboard = result['results'][0]
                print(f"🎯 First billboard confidence: {first_billboard.get('confidence', 0):.2f}")
                if first_billboard.get('detected_text'):
                    print(f"📖 Sample text: {first_billboard['detected_text'][:2]}")
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Batch processing example
    print(f"\n🔄 Batch processing example...")
    try:
        batch_results = analyzer.process_batch(example_images[:2])
        print(f"✅ Processed {len(batch_results)} images in batch")
        
        total_time = sum(r.get('processing_time', 0) for r in batch_results)
        print(f"⚡ Total batch time: {total_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Batch error: {str(e)}")
    
    print("\n🎉 Example completed!")

if __name__ == "__main__":
    main()
