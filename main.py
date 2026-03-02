from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import cv2
import numpy as np
from io import BytesIO

app = FastAPI(title="Skin Type Detection API")
# ---- Enable CORS (Allow React frontend to access API) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all frontends
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model & face detector ----
MODEL_PATH = "skin_type_model_v2.h5"
HAAR_PATH = "haarcascade_frontalface_default.xml"

model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
LABELS = ["oily", "normal", "dry"]

# ---- Helper function ----
def predict_skin_from_image(image_bytes):
    # Convert bytes to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Invalid image"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None, "No face detected"

    # Take largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face = img[y:y+h, x:x+w]

    face_resized = cv2.resize(face, (224, 224))
    face_resized = face_resized / 255.0
    face_resized = np.expand_dims(face_resized, axis=0)

    preds = model.predict(face_resized, verbose=0)
    idx = np.argmax(preds)
    label = LABELS[idx]
    confidence = float(preds[0][idx] * 100)

    return {"label": label, "confidence": confidence}, None

# ---- Routes ----
@app.get("/")
def home():
    return {"status": "running", "message": "Skin Type Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result, error = predict_skin_from_image(contents)
        if error:
            return JSONResponse({"error": error}, status_code=400)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)