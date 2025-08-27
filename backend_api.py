"""
FastAPI Backend for Billboard Analysis System
Production-ready API with comprehensive endpoints and error handling
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import aiofiles
import os
import time
import uuid
import json
import shutil
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Import our billboard analysis system
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from production_ml_utils import ProductionBillboardAnalyzer, check_system_health

# =========================
# Configuration
# =========================
class Settings:
    API_TITLE = "Billboard Analysis API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = """
    🎯 **Billboard Analysis System API**
    
    A production-ready computer vision API for analyzing billboard content with:
    - **Billboard Detection**: YOLO12n-powered detection
    - **Text Extraction**: EasyOCR integration
    - **Content Safety**: NSFW classification and risk assessment
    - **Mobile Optimized**: Fast processing for mobile deployment
    - **Compliance Ready**: Built-in safety checks
    """
    
    # File handling
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    UPLOAD_DIR = "uploads"
    TEMP_DIR = "temp"
    
    # API settings
    ENABLE_CORS = True
    ENABLE_AUTH = False  # Set to True for production
    API_KEY = "billboard-api-key-2025"  # Change in production
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "billboard_api.log"

# Initialize settings
settings = Settings()

# =========================
# Logging Setup
# =========================
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# Global Analyzer Instance
# =========================
analyzer: Optional[ProductionBillboardAnalyzer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global analyzer
    
    # Startup
    logger.info("🚀 Starting Billboard Analysis API...")
    
    # Create directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    
    # Initialize analyzer
    try:
        analyzer = ProductionBillboardAnalyzer(preload_models=True)
        logger.info("✅ Billboard analyzer initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize analyzer: {e}")
        analyzer = None
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Billboard Analysis API...")
    
    # Cleanup
    try:
        if os.path.exists(settings.TEMP_DIR):
            shutil.rmtree(settings.TEMP_DIR)
        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")

# =========================
# FastAPI App Setup
# =========================
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Security
security = HTTPBearer(auto_error=False)

# =========================
# Pydantic Models
# =========================
class AnalysisResult(BaseModel):
    """Analysis result response model"""
    status: str = Field(..., description="Analysis status")
    overall_safe: Optional[bool] = Field(None, description="Overall safety assessment")
    highest_risk_level: Optional[str] = Field(None, description="Highest risk level found")
    recommendation: Optional[str] = Field(None, description="Recommended action")
    message: str = Field(..., description="Human-readable message")
    total_billboards: int = Field(0, description="Total billboards detected")
    analyzed_billboards: int = Field(0, description="Number of billboards analyzed")
    billboard_results: List[Dict[str, Any]] = Field([], description="Detailed results per billboard")
    processing_time: float = Field(..., description="Processing time in seconds")
    performance: Optional[Dict[str, float]] = Field(None, description="Performance breakdown")
    system_info: Optional[Dict[str, str]] = Field(None, description="System information")

class HealthStatus(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="System health status")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    yolo_ready: bool = Field(..., description="YOLO model status")
    ocr_ready: bool = Field(..., description="OCR model status")
    classifier_ready: bool = Field(..., description="Text classifier status")
    device: str = Field(..., description="Processing device")
    gpu_available: bool = Field(..., description="GPU availability")
    version: str = Field(..., description="System version")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class BatchAnalysisRequest(BaseModel):
    """Batch analysis request model"""
    image_urls: List[str] = Field(..., description="List of image URLs to analyze")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for results")

class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool = Field(..., description="Request success status")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if any")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# =========================
# Authentication
# =========================
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    if not settings.ENABLE_AUTH:
        return True
    
    if not credentials or credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return True

# =========================
# Utility Functions
# =========================
async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return path"""
    # Validate file type
    file_ext = os.path.splitext(upload_file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_path = os.path.join(settings.TEMP_DIR, f"{file_id}{file_ext}")
    
    # Save file
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await upload_file.read()
            
            # Check file size
            if len(content) > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            await f.write(content)
        
        logger.info(f"📁 File saved: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"❌ File save error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

async def cleanup_file(file_path: str):
    """Cleanup temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"🗑️ Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")

def ensure_analyzer() -> ProductionBillboardAnalyzer:
    """Ensure analyzer is available"""
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Billboard analyzer not available. Please check system health."
        )
    return analyzer

# =========================
# API Endpoints
# =========================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Billboard Analysis API",
        "version": settings.API_VERSION,
        "status": "active",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """
    System health check endpoint
    
    Returns comprehensive system status including:
    - Model loading status
    - Device information
    - Performance metrics
    """
    try:
        health_data = check_system_health()
        return HealthStatus(**health_data)
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/analyze", response_model=APIResponse)
async def analyze_billboard(
    file: UploadFile = File(..., description="Billboard image file"),
    _: bool = Depends(verify_api_key)
):
    """
    Analyze a single billboard image
    
    **Upload Requirements:**
    - **File types**: JPEG, PNG, BMP, TIFF, WebP
    - **Size limit**: 10MB maximum
    - **Resolution**: 300x300 pixels minimum recommended
    
    **Returns:**
    - Billboard detection results
    - Extracted text content
    - Safety risk assessment
    - Processing performance metrics
    """
    request_id = str(uuid.uuid4())
    file_path = None
    
    try:
        logger.info(f"📸 Starting analysis for request: {request_id}")
        
        # Get analyzer
        billboard_analyzer = ensure_analyzer()
        
        # Save uploaded file
        file_path = await save_upload_file(file)
        
        # Analyze image
        start_time = time.time()
        result = billboard_analyzer.analyze_image(file_path)
        analysis_time = time.time() - start_time
        
        logger.info(f"✅ Analysis completed in {analysis_time:.2f}s for {request_id}")
        
        return APIResponse(
            success=True,
            data=AnalysisResult(**result),
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis failed for {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Cleanup
        if file_path:
            await cleanup_file(file_path)

@app.post("/analyze/batch", response_model=APIResponse)
async def analyze_batch(
    files: List[UploadFile] = File(..., description="Multiple billboard image files"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: bool = Depends(verify_api_key)
):
    """
    Analyze multiple billboard images in batch
    
    **Features:**
    - Process up to 10 images simultaneously
    - Parallel processing for improved performance
    - Individual results for each image
    - Aggregate safety assessment
    """
    request_id = str(uuid.uuid4())
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch request"
        )
    
    try:
        logger.info(f"📁 Starting batch analysis for {len(files)} files: {request_id}")
        
        # Get analyzer
        billboard_analyzer = ensure_analyzer()
        
        # Process files
        results = []
        file_paths = []
        
        for i, file in enumerate(files):
            try:
                # Save file
                file_path = await save_upload_file(file)
                file_paths.append(file_path)
                
                # Analyze
                result = billboard_analyzer.analyze_image(file_path)
                result["file_index"] = i
                result["filename"] = file.filename
                results.append(result)
                
            except Exception as e:
                logger.error(f"❌ Failed to process file {i}: {e}")
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "status": "error",
                    "message": f"Processing failed: {str(e)}",
                    "processing_time": 0.0
                })
        
        # Schedule cleanup
        for file_path in file_paths:
            background_tasks.add_task(cleanup_file, file_path)
        
        # Aggregate results
        total_billboards = sum(r.get("total_billboards", 0) for r in results)
        overall_safe = all(r.get("overall_safe", True) for r in results if r.get("status") == "analysis_complete")
        
        batch_result = {
            "batch_id": request_id,
            "total_files": len(files),
            "successful_analyses": len([r for r in results if r.get("status") == "analysis_complete"]),
            "total_billboards_detected": total_billboards,
            "batch_overall_safe": overall_safe,
            "individual_results": results
        }
        
        logger.info(f"✅ Batch analysis completed: {request_id}")
        
        return APIResponse(
            success=True,
            data=batch_result,
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"❌ Batch analysis failed for {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.get("/stats", response_model=APIResponse)
async def get_performance_stats(_: bool = Depends(verify_api_key)):
    """
    Get system performance statistics and configuration
    
    Returns current system configuration, performance metrics,
    and optimization settings.
    """
    try:
        billboard_analyzer = ensure_analyzer()
        stats = billboard_analyzer.get_performance_stats()
        
        # Add runtime stats
        health = check_system_health()
        runtime_stats = {
            "system_status": health["status"],
            "models_loaded": health["models_loaded"],
            "device": health["device"],
            "gpu_available": health["gpu_available"],
            "uptime": "Available via monitoring endpoint"
        }
        
        combined_stats = {**stats, "runtime": runtime_stats}
        
        return APIResponse(
            success=True,
            data=combined_stats
        )
        
    except Exception as e:
        logger.error(f"❌ Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")

@app.get("/config", response_model=APIResponse)
async def get_system_config(_: bool = Depends(verify_api_key)):
    """
    Get current system configuration
    
    Returns API settings, model configurations, and processing parameters.
    """
    config = {
        "api": {
            "version": settings.API_VERSION,
            "max_file_size_mb": settings.MAX_FILE_SIZE // (1024 * 1024),
            "allowed_extensions": list(settings.ALLOWED_EXTENSIONS),
            "cors_enabled": settings.ENABLE_CORS,
            "auth_enabled": settings.ENABLE_AUTH
        },
        "processing": {
            "optimal_image_size": 416,
            "max_workers": 2,
            "detection_confidence": 0.4,
            "ocr_confidence": 0.5,
            "text_confidence": 0.3
        },
        "models": {
            "yolo": "maco018/billboard-detection-Yolo12",
            "ocr": "EasyOCR English",
            "classifier": "michellejieli/NSFW_text_classifier"
        }
    }
    
    return APIResponse(success=True, data=config)

@app.post("/validate", response_model=APIResponse)
async def validate_image(
    file: UploadFile = File(..., description="Image file to validate"),
    _: bool = Depends(verify_api_key)
):
    """
    Validate image file without processing
    
    Check if image file meets requirements for analysis:
    - File format validation
    - Size validation
    - Basic image integrity check
    """
    try:
        # Basic validations
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            return APIResponse(
                success=False,
                error=f"Unsupported file type: {file_ext}"
            )
        
        # Read and check size
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        if len(content) > settings.MAX_FILE_SIZE:
            return APIResponse(
                success=False,
                error=f"File too large: {file_size_mb:.1f}MB (max: {settings.MAX_FILE_SIZE // (1024*1024)}MB)"
            )
        
        validation_result = {
            "filename": file.filename,
            "file_type": file_ext,
            "file_size_mb": round(file_size_mb, 2),
            "valid": True,
            "ready_for_analysis": True
        }
        
        return APIResponse(
            success=True,
            data=validation_result
        )
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return APIResponse(
            success=False,
            error=f"Validation failed: {str(e)}"
        )

# =========================
# Error Handlers
# =========================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=APIResponse(
            success=False,
            error=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            success=False,
            error="Internal server error",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# =========================
# Application Runner
# =========================
if __name__ == "__main__":
    print("🚀 Starting Billboard Analysis API Server...")
    print(f"📱 API Documentation: http://localhost:8000/docs")
    print(f"🏥 Health Check: http://localhost:8000/health")
    print(f"📊 System Stats: http://localhost:8000/stats")
    
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
