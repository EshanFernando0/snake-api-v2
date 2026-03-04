import os, json
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

app = FastAPI(title="Nexora Snake API v2")

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

BINARY_MODEL_PATH  = os.path.join(MODEL_DIR, "binary_best.keras")
SPECIES_MODEL_PATH = os.path.join(MODEL_DIR, "species_best.keras")
CLASS_IDX_PATH     = os.path.join(MODEL_DIR, "class_index.json")

IMG_SIZE = (224, 224)
BINARY_THRESHOLD = 0.50

# ---------- Load once on startup ----------
model_binary = tf.keras.models.load_model(BINARY_MODEL_PATH)
model_species = tf.keras.models.load_model(SPECIES_MODEL_PATH)
with open(CLASS_IDX_PATH, "r") as f:
    idx_to_class = json.load(f)

def safe_grabcut(img_bgr):
    """Try GrabCut segmentation; if it fails, return original image."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    margin = int(min(h, w) * 0.08)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask_bin = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        segmented = img_bgr * mask_bin[:, :, np.newaxis]

        # If almost empty segmentation, fallback
        if mask_bin.mean() < 0.01:
            return img_bgr
        return segmented
    except cv2.error:
        return img_bgr

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

    # JSON keys may be strings
    if isinstance(next(iter(idx_to_class.keys())), str):
        name = idx_to_class[str(idx)]
    else:
        name = idx_to_class[idx]

    return name, float(probs[idx])

@app.get("/")
def root():
    return {"status": "OK", "message": "Nexora Snake API v2 is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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