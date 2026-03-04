import os
import json
import numpy as np
import cv2
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess


app = FastAPI(title="Nexora Snake API v2")


# ----------------------------
# Paths / Settings
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

BINARY_MODEL_PATH  = os.path.join(MODEL_DIR, "binary_best.keras")
SPECIES_MODEL_PATH = os.path.join(MODEL_DIR, "species_best.keras")
CLASS_IDX_PATH     = os.path.join(MODEL_DIR, "class_index.json")

IMG_SIZE = (224, 224)
BINARY_THRESHOLD = 0.50


# ----------------------------
# Lazy model loading (fast startup)
# ----------------------------
model_binary = None
model_species = None
idx_to_class = None

def load_models_once():
    """
    Loads models only when the first request arrives.
    This avoids Azure container startup timeouts.
    """
    global model_binary, model_species, idx_to_class

    if model_binary is not None and model_species is not None and idx_to_class is not None:
        return

    # Validate files exist (better error message)
    if not os.path.exists(BINARY_MODEL_PATH):
        raise FileNotFoundError(f"Binary model not found: {BINARY_MODEL_PATH}")
    if not os.path.exists(SPECIES_MODEL_PATH):
        raise FileNotFoundError(f"Species model not found: {SPECIES_MODEL_PATH}")
    if not os.path.exists(CLASS_IDX_PATH):
        raise FileNotFoundError(f"Class index not found: {CLASS_IDX_PATH}")

    model_binary = tf.keras.models.load_model(BINARY_MODEL_PATH)
    model_species = tf.keras.models.load_model(SPECIES_MODEL_PATH)

    with open(CLASS_IDX_PATH, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)


# ----------------------------
# Segmentation (safe)
# ----------------------------
def safe_grabcut(img_bgr, iterations=5):
    """
    Try GrabCut segmentation; if it fails, return original image.
    This never crashes the API.
    """
    h, w = img_bgr.shape[:2]
    if h < 10 or w < 10:
        return img_bgr

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    margin = int(min(h, w) * 0.08)
    rect = (margin, margin, max(1, w - 2 * margin), max(1, h - 2 * margin))

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        mask_bin = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        segmented = img_bgr * mask_bin[:, :, np.newaxis]

        # If segmentation is almost empty, fallback
        if mask_bin.mean() < 0.01:
            return img_bgr

        return segmented

    except cv2.error:
        return img_bgr


# ----------------------------
# Prediction helpers
# ----------------------------
def predict_binary(img_bgr):
    img = cv2.resize(img_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # 0-255
    x = mobilenet_preprocess(img_rgb)
    x = np.expand_dims(x, 0)
    return float(model_binary.predict(x, verbose=0)[0][0])


def predict_species(img_bgr):
    img = cv2.resize(img_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # 0-255
    x = effnet_preprocess(img_rgb)
    x = np.expand_dims(x, 0)

    probs = model_species.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))

    # idx_to_class loaded from json => keys may be strings
    if isinstance(next(iter(idx_to_class.keys())), str):
        name = idx_to_class[str(idx)]
    else:
        name = idx_to_class[idx]

    return name, float(probs[idx])


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"status": "OK", "message": "Nexora Snake API v2 is running"}


@app.get("/health")
def health():
    """
    Quick health endpoint. Does NOT load models.
    Useful to see if app is alive.
    """
    return {"status": "OK", "service": "alive"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load models only when first prediction request comes
        load_models_once()

        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse({"status": "ERROR", "reason": "Invalid image"}, status_code=400)

        # Phase 1: Binary gate
        p_snake = predict_binary(img_bgr)
        if p_snake < BINARY_THRESHOLD:
            return {"status": "REJECTED", "reason": "No snake detected", "snake_prob": p_snake}

        # Phase 2: Segmentation (internal, safe)
        seg = safe_grabcut(img_bgr)

        # Phase 3: Species prediction
        species, conf = predict_species(seg)

        return {"status": "OK", "species": species, "confidence": conf, "snake_prob": p_snake}

    except FileNotFoundError as e:
        return JSONResponse({"status": "ERROR", "reason": str(e)}, status_code=500)
    except Exception as e:
        # Generic fail-safe for any unexpected error
        return JSONResponse({"status": "ERROR", "reason": f"Server error: {type(e).__name__}: {e}"}, status_code=500)
