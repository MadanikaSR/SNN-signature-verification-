import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router
from app.services.inference import inference_service

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model on startup; release resources on shutdown."""
    logger.info("Starting up — loading inference model...")
    inference_service.load()
    if inference_service.is_loaded:
        logger.info("✅ Model loaded successfully. API is ready.")
    else:
        logger.warning("⚠️  Model failed to load. /verify will return 503.")
    yield
    logger.info("Shutting down.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Signature Verification API",
    description=(
        "Siamese Neural Network-based signature verification service. "
        "Compares two signature images and returns a similarity score and match decision."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# ⚠️  In production: set ALLOWED_ORIGINS in .env to your exact Vercel domain.
#    Do NOT use "*" in production — it bypasses all cross-origin security.
allowed_origins = settings.allowed_origins_list
logger.info(f"CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────
# NOTE: No StaticFiles mount — frontend is served by Vercel independently.
app.include_router(router)


# ─── Root redirect to docs ───────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Signature Verification API", "docs": "/docs", "health": "/health"}
