#!/usr/bin/env python3
"""
Test script for bill 3.jpeg
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from production_ml_utils import ProductionBillboardAnalyzer

def test_bill_3():
    print("Testing Billboard Analysis System with bill 3.jpeg")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ProductionBillboardAnalyzer()
    
    # Test with bill 3.jpeg
    image_path = "examples/bill 3.jpeg"
    
    if os.path.exists(image_path):
        print(f"Analyzing: {image_path}")
        result = analyzer.analyze_image(image_path)
        
        print(f"\nAnalysis Results:")
        print(f"Status: {result.get('status')}")
        print(f"Billboards detected: {result.get('total_billboards', 0)}")
        print(f"Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"Overall safe: {result.get('overall_safe', 'unknown')}")
        print(f"Risk level: {result.get('highest_risk_level', 'unknown')}")
        print(f"Recommendation: {result.get('recommendation', 'unknown')}")
        
        if result.get('billboard_results'):
            print(f"\nBillboard Details:")
            for i, billboard in enumerate(result['billboard_results'], 1):
                print(f"  Billboard {i}:")
                print(f"    Confidence: {billboard.get('detection_confidence', 0):.1f}%")
                print(f"    Text: \"{billboard.get('extracted_text', 'No text')}\"")
                print(f"    Safety: {billboard.get('safety_analysis', {}).get('risk_level', 'unknown')}")
        
        return result
    else:
        print(f"Image not found: {image_path}")
        return None

if __name__ == "__main__":
    test_bill_3()
