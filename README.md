# SignVerify AI — Signature Verification System

<!-- TODO: Add a live demo badge/screenshot here after deployment -->

AI-powered signature verification using a **Siamese Neural Network** (TensorFlow) with a **FastAPI backend** deployed on **Render** and a **static frontend** deployed on **Vercel**.

<!-- TODO: Replace the placeholder URLs below with actual deployed URLs after deployment -->
<!-- 🔗 Live Demo: https://YOUR_APP.vercel.app (add link after Vercel deploy) -->
<!-- 🔗 API Docs:  https://YOUR_APP.onrender.com/docs (add link after Render deploy) -->

---

## 🚀 Live Links

| Service | URL |
|---|---|
| Frontend (Vercel) | <!-- TODO: Add Vercel URL after deployment, e.g. https://snn-signature-verification.vercel.app --> |
| Backend API (Render) | <!-- TODO: Add Render URL after deployment, e.g. https://signature-verification-api.onrender.com --> |
| API Swagger Docs | <!-- TODO: Add Render URL + /docs, e.g. https://signature-verification-api.onrender.com/docs --> |

---

## Architecture

```
[User Browser]
      │  HTTPS
      ▼
[Vercel — Static Frontend]  ──── HTTPS API ────►  [Render — FastAPI + Docker]
  frontend/index.html                              GET  /health
  frontend/style.css                               POST /verify
  frontend/app.js                                  0.0.0.0:8000
```

---

## Project Structure

```
signature-verification/
├── app/
│   ├── main.py                 # FastAPI entrypoint + CORS
│   ├── api/routes.py           # GET /health, POST /verify
│   ├── core/config.py          # pydantic-settings (extra="ignore")
│   ├── services/inference.py   # singleton ML inference
│   ├── utils/preprocessing.py  # bytes → numpy
│   └── models/loader.py        # .h5 loader
├── ml/
│   └── train.py                # Siamese network training on CEDAR
├── frontend/
│   ├── index.html              # standalone SaaS UI
│   ├── style.css               # glassmorphism dark theme
│   ├── app.js                  # env-aware API calls + cold-start retry
│   └── vercel.json             # Vercel static routing + security headers
├── models/                     # .h5 files — included in repo for easy Docker deployment
├── render.yaml                 # Render infrastructure-as-code config
├── .env.example                # Environment variable template
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Local Development

### Prerequisites
- Python 3.10+
- CEDAR dataset (for training only) — place at `./CEDAR/`

### 1. Backend

```bash
# Clone & create virtualenv
git clone https://github.com/MadanikaSR/SNN-signature-verification-.git
cd SNN-signature-verification-
python -m venv venv

# Windows
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env template
cp .env.example .env
# Edit .env if needed (defaults work for local dev)

# (Optional) Train the model — requires CEDAR dataset
python -m ml.train

# Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### 2. Frontend

Open `frontend/index.html` directly in a browser, **or** serve locally:

```bash
cd frontend
python -m http.server 3000
# Open: http://localhost:3000
```

The `app.js` automatically detects `localhost` and points to `http://localhost:8000` — no config needed.

---

## Docker

```bash
# Build & start
docker-compose up --build

# Test
curl http://localhost:8000/health
```

Place your `.h5` model files in `./models/` before starting — they are mounted as a read-only volume.

---

## Deployment

### Backend → Render

> The `render.yaml` in the repo root pre-configures most settings.

1. Push this repo to GitHub (already done if you're reading this)
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect this GitHub repo
4. Configure:
   - **Runtime**: Docker
   - **Dockerfile Path**: `./Dockerfile`
   - **Instance Type**: Free
5. Set these **Environment Variables** in the Render dashboard:

   ```
   MODEL_PATH    = models/siamese_signature_model.h5
   ENCODER_PATH  = models/signature_encoder.h5
   MATCH_THRESHOLD = 0.5
   IMG_SIZE      = 128
   MAX_FILE_SIZE_MB = 2
   ALLOWED_ORIGINS = https://YOUR_VERCEL_APP.vercel.app  <!-- TODO: update after Vercel deploy -->
   ```

6. Render will automatically pull the models from the repository and build the Docker image. Keep an eye on the deployment logs for `✅ Model loaded successfully.`

<!-- TODO: Add your Render URL here after deployment -->
<!-- Render URL: https://YOUR_SERVICE.onrender.com -->

---

### Frontend → Vercel

1. In `frontend/app.js` (line 13), replace the production URL placeholder:
   ```js
   // TODO: Replace with your actual Render URL after Step 8 above
   : "https://YOUR_RENDER_BACKEND_URL.onrender.com";
   ```

2. Commit and push:
   ```bash
   git add frontend/app.js
   git commit -m "Set production Render API URL"
   git push
   ```

3. Go to [vercel.com](https://vercel.com) → **New Project** → import this repo
4. Set **Root Directory** to `frontend` ← important
5. Leave **Build Command** and **Output Directory** empty
6. Click **Deploy**

<!-- TODO: Add your Vercel URL here after deployment -->
<!-- Vercel URL: https://YOUR_APP.vercel.app -->

7. Go back to **Render → Environment Variables** → update `ALLOWED_ORIGINS` to your Vercel URL and redeploy.

---

## Model Files

The trained `.h5` model files (~6 MB total) are included directly in the repository. Docker and Render will automatically load them on startup without any extra configuration.

---

## API Reference

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime_seconds": 123.4
}
```

### `POST /verify`

**Request**: `multipart/form-data`

| Field  | Type | Description |
|--------|------|-------------|
| `image1` | File | Reference signature (PNG/JPG, max 2 MB) |
| `image2` | File | Query signature (PNG/JPG, max 2 MB) |

**Response**:
```json
{
  "similarity_score": 0.8723,
  "is_match": true,
  "confidence": 74.46,
  "threshold": 0.5
}
```

**Error codes**:
- `415` — unsupported file type (only PNG/JPG accepted)
- `413` — file too large (> 2 MB)
- `503` — model not loaded (check Render disk + restart)

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/siamese_signature_model.h5` | Path to Siamese model |
| `ENCODER_PATH` | `models/signature_encoder.h5` | Path to encoder model |
| `MATCH_THRESHOLD` | `0.5` | Similarity cutoff for match decision |
| `IMG_SIZE` | `128` | Image resize dimension (px) |
| `MAX_FILE_SIZE_MB` | `2` | Max upload file size |
| `ALLOWED_ORIGINS` | `http://localhost:3000,...` | CORS origins (comma-separated) |
| `CEDAR_ROOT` | `./CEDAR` | CEDAR dataset path (training only) |
| `TRAIN_EPOCHS` | `12` | Training epochs (training only) |

Training-only variables (`CEDAR_ROOT`, `TRAIN_EPOCHS`, `TRAIN_BATCH_SIZE`, `TRAIN_STEPS_PER_EPOCH`, `TRAIN_VALIDATION_STEPS`) are safely ignored by the API server.

---

## Security Notes

- **CORS**: Never use `"*"` in production. Set `ALLOWED_ORIGINS` to your exact Vercel URL.
- **Models**: `.h5` files are included directly in the repo for easy deployment on the Free tier.
- **Secrets**: Never commit `.env`. Only `.env.example` is tracked.

---

## Training Results

Trained on the [CEDAR signature dataset](http://www.cedar.buffalo.edu/NIJ/data/) (55 subjects, 24 originals + 24 forgeries each).

| Epoch | Train Acc | Val Acc | Val Loss |
|-------|-----------|---------|----------|
| 1 | 69.5% | 75.8% | 0.661 |
| 6 | 96.6% | 97.4% | 0.317 |
| 10 | 98.5% | 99.2% | 0.167 |
| **12** | **98.8%** | **99.1%** | **0.155** |
