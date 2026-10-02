# FastAPI Backend Documentation
## Billboard Analysis System

### Overview
This FastAPI backend provides a production-ready REST API for the Billboard Analysis System, offering comprehensive billboard detection, text extraction, and content safety analysis.

---

## 🚀 Quick Start

### Local Development
```bash
# Install backend dependencies
pip install -r requirements-backend.txt

# Start the FastAPI server
python backend_api.py

# API will be available at:
# - Main API: http://localhost:8000
# - Interactive docs: http://localhost:8000/docs
# - ReDoc docs: http://localhost:8000/redoc
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build Docker image manually
docker build -t billboard-api .
docker run -p 8000:8000 billboard-api
```

---

## 📋 API Endpoints

### 🏥 Health & Status

#### `GET /health`
Check system health and model status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "yolo_ready": true,
  "ocr_ready": true,
  "classifier_ready": true,
  "device": "cpu",
  "gpu_available": false,
  "version": "1.0",
  "timestamp": "2025-08-25T10:30:00Z"
}
```

#### `GET /stats`
Get performance statistics and configuration.

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization_level": "production",
    "device": "cpu",
    "image_size": 416,
    "workers": 2,
    "expected_performance": "50-65% faster than baseline",
    "runtime": {
      "system_status": "healthy",
      "models_loaded": true
    }
  }
}
```

#### `GET /config`
Get current system configuration.

**Response:**
```json
{
  "success": true,
  "data": {
    "api": {
      "version": "1.0.0",
      "max_file_size_mb": 10,
      "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    },
    "processing": {
      "optimal_image_size": 416,
      "detection_confidence": 0.4,
      "ocr_confidence": 0.5
    },
    "models": {
      "yolo": "maco018/billboard-detection-Yolo12",
      "ocr": "EasyOCR English",
      "classifier": "michellejieli/NSFW_text_classifier"
    }
  }
}
```

### 📸 Image Analysis

#### `POST /analyze`
Analyze a single billboard image.

**Request:**
- **Content-Type**: `multipart/form-data`
- **File**: Image file (JPEG, PNG, etc.)
- **Max Size**: 10MB

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@billboard.jpg"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "analysis_complete",
    "overall_safe": false,
    "highest_risk_level": "medium_risk",
    "recommendation": "review",
    "message": "Moderate risk detected - manual review recommended",
    "total_billboards": 1,
    "analyzed_billboards": 1,
    "billboard_results": [
      {
        "billboard_id": 1,
        "bbox": [482, 66, 1024, 768],
        "detection_confidence": 85.77,
        "extracted_text": "Sample billboard text content",
        "text_confidence": 0.641,
        "safety_analysis": {
          "safe": false,
          "confidence": 0.658,
          "label": "NSFW",
          "risk_level": "medium_risk",
          "action": "review"
        }
      }
    ],
    "processing_time": 3.62,
    "performance": {
      "detection_time": 0.679,
      "ocr_time": 2.631,
      "analysis_time": 0.972
    }
  },
  "timestamp": "2025-08-25T10:30:00Z",
  "request_id": "req_123456"
}
```

#### `POST /analyze/batch`
Analyze multiple billboard images in batch.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Files**: Multiple image files (max 10)

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/analyze/batch" \
  -F "files=@billboard1.jpg" \
  -F "files=@billboard2.jpg" \
  -F "files=@billboard3.jpg"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_123456",
    "total_files": 3,
    "successful_analyses": 3,
    "total_billboards_detected": 5,
    "batch_overall_safe": true,
    "individual_results": [
      {
        "file_index": 0,
        "filename": "billboard1.jpg",
        "status": "analysis_complete",
        "total_billboards": 2,
        "recommendation": "approve"
      }
    ]
  }
}
```

#### `POST /validate`
Validate image file without processing.

**Request:**
- **Content-Type**: `multipart/form-data`
- **File**: Image file to validate

**Response:**
```json
{
  "success": true,
  "data": {
    "filename": "billboard.jpg",
    "file_type": ".jpg",
    "file_size_mb": 2.5,
    "valid": true,
    "ready_for_analysis": true
  }
}
```

---

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_TITLE="Billboard Analysis API"
API_VERSION="1.0.0"
MAX_FILE_SIZE=10485760  # 10MB in bytes
UPLOAD_DIR="uploads"

# Security
ENABLE_AUTH=false
API_KEY="your-secure-api-key"

# Logging
LOG_LEVEL="INFO"
LOG_FILE="billboard_api.log"

# Processing
ENABLE_GPU=false
OPTIMAL_SIZE=416
MAX_WORKERS=2
```

### File Requirements
- **Supported formats**: JPEG, PNG, BMP, TIFF, WebP
- **Size limit**: 10MB maximum
- **Resolution**: 300x300 pixels minimum recommended
- **Color space**: RGB or sRGB

---

## 🛡️ Security Features

### Authentication
- **Bearer Token**: Optional API key authentication
- **File Validation**: Comprehensive file type and size validation
- **Input Sanitization**: Protection against malicious uploads
- **Rate Limiting**: Configurable request rate limits

### Data Protection
- **No Data Retention**: Images processed and immediately deleted
- **Local Processing**: No external API calls after initialization
- **Secure Uploads**: Temporary file handling with automatic cleanup
- **Privacy**: No personal data storage or logging

---

## 📊 Response Format

### Standard API Response
```json
{
  "success": boolean,
  "data": object | null,
  "error": string | null,
  "timestamp": "ISO 8601 datetime",
  "request_id": "unique identifier"
}
```

### Error Responses
```json
{
  "success": false,
  "error": "Detailed error message",
  "timestamp": "2025-08-25T10:30:00Z",
  "request_id": "req_123456"
}
```

### Status Codes
- `200` - Success
- `400` - Bad Request (invalid file, size limit)
- `401` - Unauthorized (invalid API key)
- `413` - Payload Too Large
- `422` - Validation Error
- `500` - Internal Server Error
- `503` - Service Unavailable (models not loaded)

---

## 🚀 Performance Optimization

### Processing Performance
- **Average response time**: 2-5 seconds per image
- **Batch processing**: Parallel analysis support
- **Memory optimization**: Automatic cleanup and garbage collection
- **Caching**: Model preloading and intelligent caching

### Scaling Recommendations
```yaml
Production_Setup:
  instances: 3-5 API instances
  load_balancer: Nginx or AWS ALB
  caching: Redis for session management
  monitoring: Prometheus + Grafana
  
Resource_Requirements:
  cpu: 4+ cores per instance
  memory: 4GB+ per instance
  storage: 2GB for models and cache
  network: High bandwidth for image uploads
```

---

## 🧪 Testing

### Test Client Usage
```python
from test_api_client import BillboardAPIClient

# Initialize client
client = BillboardAPIClient("http://localhost:8000")

# Test health
health = client.health_check()
print(health)

# Analyze image
result = client.analyze_image("path/to/billboard.jpg")
print(result)

# Batch analysis
results = client.analyze_batch(["img1.jpg", "img2.jpg"])
print(results)
```

### Manual Testing
```bash
# Run test client
python test_api_client.py

# Test with curl
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@examples/img1.jpeg"

# Health check
curl http://localhost:8000/health
```

---

## 🔍 Monitoring & Logging

### Log Levels
- **INFO**: Normal operation logs
- **WARNING**: Performance issues or recoverable errors
- **ERROR**: Failed requests or system errors
- **DEBUG**: Detailed debugging information

### Metrics Available
- Request count and response times
- Model performance statistics
- Error rates and types
- Resource utilization
- Cache hit rates

### Health Monitoring
```bash
# Check API health
curl http://localhost:8000/health

# Check system stats
curl http://localhost:8000/stats

# Validate system config
curl http://localhost:8000/config
```

---

## 🚀 Production Deployment

### AWS Deployment
```yaml
ECS_Service:
  - Task Definition with Docker image
  - Application Load Balancer
  - Auto Scaling Group
  - CloudWatch monitoring

Lambda_Alternative:
  - Serverless deployment option
  - API Gateway integration
  - S3 for temporary file storage
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: billboard-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: billboard-api
  template:
    metadata:
      labels:
        app: billboard-api
    spec:
      containers:
      - name: api
        image: billboard-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

This comprehensive FastAPI backend provides enterprise-grade API functionality for your Billboard Analysis System with full production readiness!
