# API Documentation

## BillboardAnalyzer Class

The main class for billboard detection and analysis.

### Initialization

```python
analyzer = BillboardAnalyzer()
```

### Methods

#### `analyze_image(image_path)`

Analyzes a single image for billboard content.

**Parameters:**
- `image_path` (str): Path to the image file

**Returns:**
- `dict`: Analysis results with the following structure:

```json
{
    "status": "success|error",
    "processing_time": float,
    "billboards_detected": int,
    "total_text_blocks": int,
    "vulnerability_score": float,
    "results": [
        {
            "billboard_id": int,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
            "detected_text": ["text1", "text2"],
            "vulnerability_analysis": {
                "is_nsfw": bool,
                "confidence": float
            }
        }
    ]
}
```

#### `process_batch(image_paths)`

Processes multiple images in batch.

**Parameters:**
- `image_paths` (list): List of image file paths

**Returns:**
- `list`: List of analysis results for each image

#### `health_check()`

Checks system health and model availability.

**Returns:**
- `dict`: System status information

```json
{
    "status": "healthy|error",
    "models_loaded": bool,
    "yolo_model": "available|error",
    "ocr_model": "available|error", 
    "nsfw_model": "available|error",
    "memory_usage": float,
    "cache_size": int
}
```

#### `get_performance_stats()`

Returns performance statistics.

**Returns:**
- `dict`: Performance metrics

```json
{
    "total_images_processed": int,
    "average_processing_time": float,
    "success_rate": float,
    "cache_hit_rate": float
}
```

## Configuration

### Environment Variables

- `BILLBOARD_CACHE_DIR`: Custom cache directory (default: './cache')
- `BILLBOARD_MODEL_DIR`: Custom model directory (default: './models')
- `BILLBOARD_LOG_LEVEL`: Logging level (default: 'INFO')

### Model Configuration

The system automatically downloads and caches models:
- YOLO12n: `maco018/billboard-detection-Yolo12`
- NSFW Classifier: `michellejieli/NSFW_text_classifier`
- EasyOCR: English language pack

## Error Handling

All methods return structured error responses:

```json
{
    "status": "error",
    "error_type": "ModelError|ImageError|ProcessingError",
    "message": "Detailed error message",
    "traceback": "Full traceback for debugging"
}
```

## Performance Optimization

### Speed Optimizations
- Model preloading and caching
- Image resizing to optimal dimensions
- Batch processing capabilities
- Multi-threading support

### Memory Management
- Automatic cache cleanup
- Model unloading for mobile environments
- Memory usage monitoring

## Mobile Deployment

For mobile environments, use these optimizations:

```python
# Enable mobile mode
analyzer = BillboardAnalyzer(mobile_mode=True)

# Process with memory constraints
result = analyzer.analyze_image(
    image_path, 
    max_size=416,  # Smaller image size
    enable_gpu=False  # CPU-only processing
)
```
