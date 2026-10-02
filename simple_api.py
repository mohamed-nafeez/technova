"""
Simple FastAPI for testing file uploads without heavy ML model loading
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
import shutil

app = FastAPI(title="Simple Billboard API", version="1.0.0")

# Create upload directory
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Simple Billboard Detection API", "status": "active"}

@app.get("/health")
def health_check():
    return JSONResponse({
        "status": "healthy",
        "service": "simple_billboard_api",
        "version": "1.0.0"
    })

@app.post("/test_upload")
def test_upload(image: UploadFile = File(...)):
    """Test file upload without ML processing"""
    
    # Create safe filename
    clean_filename = image.filename.replace('/', '_').replace('\\', '_')
    safe_filename = f"{uuid.uuid4()}_{clean_filename}"
    temp_filename = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        # Save file
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Get file info
        file_size = os.path.getsize(temp_filename)
        
        # Clean up
        os.remove(temp_filename)
        
        return JSONResponse({
            "success": True, 
            "filename": clean_filename,
            "file_size": file_size,
            "message": "File uploaded and processed successfully"
        })
    
    except Exception as e:
        # Clean up on error
        try:
            os.remove(temp_filename)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
