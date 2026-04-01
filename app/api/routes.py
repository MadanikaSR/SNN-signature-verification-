import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.services.inference import inference_service

router = APIRouter()

# Track application start time for uptime reporting
_start_time = time.time()

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # MB → bytes


# ─── Health Check ──────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """
    Render uses this endpoint for uptime monitoring.
    Returns service status and whether the ML model is loaded.
    """
    return {
        "status": "ok",
        "model_loaded": inference_service.is_loaded,
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


# ─── Signature Verification ───────────────────────────────────────────────────

@router.post("/verify")
async def verify_signatures(
    image1: UploadFile = File(..., description="Reference signature image (PNG/JPG)"),
    image2: UploadFile = File(..., description="Query signature image (PNG/JPG)"),
):
    """
    Compare two signature images using the Siamese Neural Network.

    - Accepts multipart/form-data with two image files
    - Returns similarity score, match decision, and confidence level
    """
    if not inference_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. The service may still be starting up.",
        )

    # Validate both uploads
    for upload, label in [(image1, "image1"), (image2, "image2")]:
        if upload.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"{label}: unsupported file type '{upload.content_type}'. Use PNG or JPG.",
            )

    # Read bytes
    img1_bytes = await image1.read()
    img2_bytes = await image2.read()

    # File size validation
    for raw, label in [(img1_bytes, "image1"), (img2_bytes, "image2")]:
        if len(raw) > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{label} exceeds maximum size of {settings.MAX_FILE_SIZE_MB}MB.",
            )

    try:
        similarity_score, is_match, confidence = inference_service.predict(img1_bytes, img2_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")

    return JSONResponse(content={
        "similarity_score": similarity_score,
        "is_match": is_match,
        "confidence": confidence,
        "threshold": settings.MATCH_THRESHOLD,
    })
