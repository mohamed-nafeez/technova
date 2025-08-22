# Deployment Guide

## Overview

This guide covers deploying the Billboard Analysis System in various environments.

## Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (2GB for mobile)
- Internet connection for initial model downloads

## Installation

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/billboard-analysis-system.git
cd billboard-analysis-system

# Install dependencies
pip install -r requirements.txt

# Run tests (optional)
python -m pytest tests/

# Test the system
python examples/demo.py
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from src.production_ml_utils import BillboardAnalyzer; BillboardAnalyzer()"

EXPOSE 8000
CMD ["python", "examples/demo.py"]
```

### Mobile/Edge Deployment

For resource-constrained environments:

```bash
# Install lightweight dependencies
pip install -r requirements.txt --no-deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Configure for mobile
export BILLBOARD_MOBILE_MODE=true
export BILLBOARD_MAX_IMAGE_SIZE=416
```

## Production Deployment

### Backend API Server

```python
from flask import Flask, request, jsonify
from src.production_ml_utils import BillboardAnalyzer

app = Flask(__name__)
analyzer = BillboardAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    # Save temporarily and analyze
    temp_path = f"/tmp/{image_file.filename}"
    image_file.save(temp_path)
    
    result = analyzer.analyze_image(temp_path)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(analyzer.health_check())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### Performance Tuning

#### Memory Optimization

```python
# Configure cache limits
analyzer = BillboardAnalyzer(
    cache_size_limit=1000,  # Max cached models
    memory_threshold=0.8    # Memory usage threshold
)
```

#### Speed Optimization

```python
# Batch processing for multiple images
results = analyzer.process_batch(
    image_paths, 
    batch_size=8,
    num_workers=4
)
```

## Monitoring

### Health Checks

```python
# Regular health monitoring
health = analyzer.health_check()
if health['status'] != 'healthy':
    # Alert or restart system
    pass
```

### Performance Metrics

```python
# Get performance statistics
stats = analyzer.get_performance_stats()
print(f"Average processing time: {stats['average_processing_time']}s")
print(f"Success rate: {stats['success_rate']}%")
```

## Troubleshooting

### Common Issues

1. **Model Download Failures**
   ```bash
   # Clear cache and retry
   rm -rf cache/
   python -c "from src.production_ml_utils import BillboardAnalyzer; BillboardAnalyzer()"
   ```

2. **Memory Issues**
   ```python
   # Enable mobile mode
   analyzer = BillboardAnalyzer(mobile_mode=True)
   ```

3. **Slow Processing**
   ```python
   # Optimize image size
   result = analyzer.analyze_image(image_path, max_size=416)
   ```

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Security Considerations

1. **Input Validation**: Always validate image inputs
2. **File Permissions**: Secure temporary file handling
3. **Resource Limits**: Set processing timeouts
4. **Network Security**: Use HTTPS for API endpoints

## Scaling

### Horizontal Scaling

Deploy multiple instances with a load balancer:

```yaml
# docker-compose.yml
version: '3.8'
services:
  billboard-analyzer:
    build: .
    ports:
      - "8000-8003:8000"
    environment:
      - BILLBOARD_CACHE_DIR=/shared/cache
    volumes:
      - ./cache:/shared/cache
```

### Vertical Scaling

For high-throughput scenarios:
- Use GPU acceleration
- Increase batch sizes
- Enable multi-threading
- Optimize image preprocessing
