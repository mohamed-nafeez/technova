"""
Mobile-optimized configuration for ultra-fast performance
"""

class MobileConfig:
    # Ultra-fast mobile settings
    MOBILE_SIZE = 320          # Even smaller for speed
    MOBILE_CONFIDENCE = 0.6    # Higher threshold = fewer detections
    MOBILE_MAX_DETECTIONS = 3  # Limit detections for speed
    MOBILE_WORKERS = 1         # Single thread for stability
    
def create_mobile_analyzer():
    """Create analyzer optimized for mobile"""
    
    # Override configs for mobile
    Config.OPTIMAL_SIZE = MobileConfig.MOBILE_SIZE
    Config.DETECTION_CONFIDENCE = MobileConfig.MOBILE_CONFIDENCE
    Config.MAX_WORKERS = MobileConfig.MOBILE_WORKERS
    
    analyzer = ProductionBillboardAnalyzer()
    return analyzer

# Usage for mobile
mobile_analyzer = create_mobile_analyzer()
result = mobile_analyzer.analyze_image("billboard.jpg")  # Faster on mobile
