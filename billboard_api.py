from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import shutil
import os
import uuid
import sys
import cv2
import numpy as np
from io import BytesIO
import base64

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from production_ml_utils import detect_billboards, extract_text_from_crops, analyze_text_safety
    ML_MODEL_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import ML model: {e}")
    ML_MODEL_AVAILABLE = False

app = FastAPI(title="Billboard Detection API")

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Billboard Detection API", "ml_model_available": ML_MODEL_AVAILABLE}

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if ML_MODEL_AVAILABLE else "degraded",
        "ml_model_available": ML_MODEL_AVAILABLE,
        "upload_dir_exists": os.path.exists(UPLOAD_DIR)
    }

@app.post("/detect_billboard")
def detect_billboard_api(image: UploadFile = File(...)):
    if not ML_MODEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    # Create safe filename
    clean_filename = image.filename.replace('/', '_').replace('\\', '_')
    safe_filename = f"{uuid.uuid4()}_{clean_filename}"
    temp_filename = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Run detection
        detections = detect_billboards(temp_filename)
        num_billboards = len(detections)
        
        # Clean up temp file
        try:
            os.remove(temp_filename)
        except Exception:
            pass
        
        # Business logic: Only one billboard is valid
        if num_billboards == 1:
            return JSONResponse({"billboard_detected": True, "reason": "Exactly one billboard detected."})
        elif num_billboards == 0:
            return JSONResponse({"billboard_detected": False, "reason": "No billboard detected."})
        else:
            return JSONResponse({"billboard_detected": False, "reason": f"{num_billboards} billboards detected. Only one allowed."})
    
    except Exception as e:
        # Clean up temp file on error
        try:
            os.remove(temp_filename)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/crop_billboard")
def crop_billboard_api(image: UploadFile = File(...)):
    """
    Detects billboard and returns the cropped billboard image.
    Only works if exactly one billboard is detected.
    """
    if not ML_MODEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    temp_filename = None
    try:
        # Save uploaded image to disk
        temp_filename = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{image.filename}")
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Run detection
        detections = detect_billboards(temp_filename)
        num_billboards = len(detections)
        
        # Business logic: Only process if exactly one billboard
        if num_billboards == 0:
            return JSONResponse(
                status_code=404,
                content={"error": "No billboard detected in the image"}
            )
        elif num_billboards > 1:
            return JSONResponse(
                status_code=400,
                content={"error": f"{num_billboards} billboards detected. Only one allowed for cropping."}
            )
        
        # Get the single billboard detection
        detection = detections[0]
        
        # Load original image and crop billboard
        original_img = cv2.imread(temp_filename)
        if original_img is None:
            raise HTTPException(status_code=400, detail="Could not load image")
        
        # Get bounding box coordinates
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        
        # Crop the billboard region
        cropped_billboard = original_img[y1:y2, x1:x2]
        
        # Convert cropped image to JPEG bytes
        _, img_encoded = cv2.imencode('.jpg', cropped_billboard)
        img_bytes = img_encoded.tobytes()
        
        # Clean up temp file
        try:
            os.remove(temp_filename)
        except Exception:
            pass
        
        # Return cropped image as response
        return StreamingResponse(
            BytesIO(img_bytes),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename=cropped_billboard_{uuid.uuid4().hex[:8]}.jpg"
            }
        )
    
    except Exception as e:
        # Clean up temp file on error
        if temp_filename:
            try:
                os.remove(temp_filename)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Cropping error: {str(e)}")

@app.post("/extract_text")
def extract_text_api(image: UploadFile = File(...)):
    """
    Takes a cropped billboard image as input and returns:
    - Extracted OCR text (with confidence filtering)
    - Safety level analysis of the text
    """
    if not ML_MODEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    temp_filename = None
    try:
        # Save uploaded image to disk
        temp_filename = os.path.join(UPLOAD_DIR, f"ocr_{uuid.uuid4()}_{image.filename}")
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Load image for OCR processing
        img = cv2.imread(temp_filename)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not load image")
        
        # Extract text using OCR
        ocr_results = extract_text_from_crops([img])
        
        # Clean up temp file
        try:
            os.remove(temp_filename)
        except Exception:
            pass
        
        if not ocr_results or len(ocr_results) == 0:
            return JSONResponse({
                "success": True,
                "text": "",
                "confidence": 0.0,
                "safety_level": "safe",
                "safety_confidence": 1.0,
                "message": "No text detected in the image"
            })
        
        # Get the first (and only) OCR result
        ocr_result = ocr_results[0]
        extracted_text = ocr_result.get("text", "")
        text_confidence = ocr_result.get("confidence", 0.0)
        
        # Only proceed with safety analysis if text was found and meets confidence threshold
        if not extracted_text or text_confidence < 0.3:  # Minimum confidence threshold
            return JSONResponse({
                "success": True,
                "text": "",
                "confidence": text_confidence,
                "safety_level": "safe",
                "safety_confidence": 1.0,
                "message": f"Text confidence too low ({text_confidence:.2f}) or no text found"
            })
        
        # Analyze text safety
        safety_results = analyze_text_safety([extracted_text])
        safety_analysis = safety_results[0] if safety_results else {
            "safe": True, 
            "confidence": 1.0, 
            "risk_level": "safe",
            "label": "SAFE"
        }
        
        return JSONResponse({
          #  "success": True,
            "text": extracted_text,
           # "confidence": text_confidence,
            #"word_count": ocr_result.get("word_count", 0),
            #"safety_level": safety_analysis.get("risk_level", "safe"),
            "safety_confidence": safety_analysis.get("confidence", 1.0),
           # "safety_label": safety_analysis.get("label", "SAFE"),
            #"is_safe": safety_analysis.get("safe", True),
            #"recommended_action": safety_analysis.get("action", "approve"),
            #"message": "Text extracted and analyzed successfully"
        })
    
    except Exception as e:
        # Clean up temp file on error
        if temp_filename:
            try:
                os.remove(temp_filename)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
