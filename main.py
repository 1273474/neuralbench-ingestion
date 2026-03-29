"""
NeuralBench TRIBE v2 — FastAPI Ingestion Service
═════════════════════════════════════════════════
Accepts video uploads, runs them through the TRIBE v2 trimodal
brain encoder inference pipeline, and returns predicted cognitive
load metrics derived from 70,000-voxel fMRI activation maps.
"""

import os
import tempfile
import logging
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from models.schemas import AnalysisResult
from services.video_analyzer import extract_metrics

# ── Logging Configuration ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(
    title="NeuralBench TRIBE v2 Ingestion Service",
    version="2.0.0",
    description=(
        "Ingests video media and predicts cognitive load metrics "
        "(Visual Intensity, Audio Complexity, Text Density) using the "
        "Meta TRIBE v2 Trimodal Brain Encoder foundation model."
    ),
)

# ── Upload Constraints ─────────────────────────────────────────
MAX_FILE_SIZE_MB: int = 50
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: set = {".mp4", ".mov", ".mkv", ".avi"}
SUPPORTED_DEMOGRAPHICS: set = {"Gen-Z", "Academic", "Professional"}


# ════════════════════════════════════════════════════════════════
# POST /analyze-media
# ════════════════════════════════════════════════════════════════

@app.post("/analyze-media", response_model=AnalysisResult)
async def analyze_media(
    file: UploadFile = File(..., description="Video file (.mp4, .mov, .mkv, .avi)"),
    demographic: str = Form(
        default="Gen-Z",
        description="Subject cohort for the TRIBE v2 Subject Residual Block. "
                    "One of: Gen-Z, Academic, Professional.",
    ),
):
    """
    Ingests an uploaded video file and runs it through the full
    TRIBE v2 inference pipeline:
    
    1. Trimodal encoding (vision + audio + text)
    2. Subject Residual Block biasing by demographic
    3. TRIBE decoder → 70,000 fMRI voxel prediction
    4. Voxel-to-metric dimensionality reduction
    
    **Limits:**
    - Max file size: 50MB (HTTP 413 if exceeded)
    - Max video duration: 120 seconds (HTTP 400 if exceeded)
    - Supported demographics: Gen-Z, Academic, Professional
    
    **Error codes:**
    - 400: Invalid file, unsupported format, bad demographic, or video too long
    - 413: File size exceeds 50MB
    - 503: CUDA Out-of-Memory (GPU overloaded)
    """
    # ── Validate filename ──────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # ── Validate demographic ───────────────────────────────────
    if demographic not in SUPPORTED_DEMOGRAPHICS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported demographic '{demographic}'. "
                   f"Must be one of: {', '.join(sorted(SUPPORTED_DEMOGRAPHICS))}",
        )

    # ── Read and validate file size ────────────────────────────
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB.",
        )

    logger.info(
        f"Received upload: {file.filename} ({len(file_bytes):,} bytes), "
        f"demographic={demographic}"
    )

    # ── Save to secure temp file ───────────────────────────────
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(temp_fd, "wb") as temp_file:
            temp_file.write(file_bytes)

        logger.info(f"Saved to {temp_path}. Starting TRIBE v2 inference...")

        # ── Run the TRIBE v2 inference pipeline ────────────────
        metrics = extract_metrics(temp_path, demographic=demographic)

        return AnalysisResult(**metrics)

    except ValueError as ve:
        # Validation errors from the analyzer (bad video, duration, etc.)
        logger.warning(f"Validation Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except torch.cuda.OutOfMemoryError:
        # CUDA OOM — the GPU cannot handle this workload right now.
        # Clear the cache and return a 503 so the client can retry.
        torch.cuda.empty_cache()
        logger.error(
            "CUDA Out-of-Memory during TRIBE v2 inference. "
            "GPU VRAM exhausted — consider reducing batch size or "
            "upgrading to a higher-VRAM GPU."
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "GPU memory exhausted during neural inference. "
                "The server is temporarily unable to process this request. "
                "Please retry in a few moments or contact the administrator."
            ),
        )

    except Exception as e:
        logger.error(f"Internal processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during TRIBE v2 inference.",
        )

    finally:
        # ── Always clean up temp file ──────────────────────────
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temporary file: {temp_path}")


# ════════════════════════════════════════════════════════════════
# Health Check
# ════════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    """Returns server health and device info."""
    return {
        "status": "ok",
        "model": "TRIBE v2 (Trimodal Brain Encoder)",
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "mock_inference": os.environ.get("MOCK_INFERENCE", "true").lower() == "true",
    }
