"""Local Binoculars API server for Mac Studio.

Same API contract as grant_assist/binoculars-service so it's a drop-in
replacement during development.

Usage:
    python -m binoculars_local.server
    # → http://localhost:8080/docs
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import config
from .detector import (
    MIN_TOKENS,
    THRESHOLD_ACCURACY,
    THRESHOLD_LOW_FPR,
    BinocularsDetector,
    get_device,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --- Pydantic schemas (same as binoculars-service) ---

class ChunkInput(BaseModel):
    id: str
    text: str


class DetectRequest(BaseModel):
    chunks: list[ChunkInput] = Field(..., min_length=1)
    threshold_mode: str = Field(
        default="low_fpr",
        pattern="^(accuracy|low_fpr)$",
    )


class ChunkResult(BaseModel):
    id: str
    score: float
    label: str
    token_count: int


class ModelInfo(BaseModel):
    observer: str
    performer: str
    quantization: str
    threshold_mode: str
    threshold_low_fpr: float
    threshold_accuracy: float


class DetectResponse(BaseModel):
    results: list[ChunkResult]
    model_info: ModelInfo


# --- App ---

detector = BinocularsDetector(device=get_device(config.device))


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector.load_models()
    yield


app = FastAPI(
    title="Binoculars AI Detection (Mac Studio Local)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "healthy" if detector.is_loaded else "loading",
        "models_loaded": detector.is_loaded,
        "device": str(detector.device),
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Models not yet loaded")

    results: list[ChunkResult] = []
    start = time.monotonic()

    for chunk in request.chunks:
        text = chunk.text.strip()
        if not text:
            results.append(
                ChunkResult(id=chunk.id, score=0.0, label="skipped", token_count=0)
            )
            continue

        score, token_count = detector.compute_score(text)

        if token_count < MIN_TOKENS:
            label = "uncertain"
        else:
            label = detector.classify(score, request.threshold_mode)

        results.append(
            ChunkResult(
                id=chunk.id,
                score=round(score, 4),
                label=label,
                token_count=token_count,
            )
        )

    elapsed = time.monotonic() - start
    logger.info("Processed %d chunks in %.1fs", len(request.chunks), elapsed)

    return DetectResponse(
        results=results,
        model_info=ModelInfo(
            observer=config.observer_model,
            performer=config.performer_model,
            quantization="none (fp16 on MPS)",
            threshold_mode=request.threshold_mode,
            threshold_low_fpr=THRESHOLD_LOW_FPR,
            threshold_accuracy=THRESHOLD_ACCURACY,
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
