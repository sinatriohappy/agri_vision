"""
Agri Vision — Flask Application
================================
Serves the web UI and handles image-based rice disease prediction.

Capstone Project by:
  Stefanus Westin G.M.  (71241073)
  Giovanni Albert Harjanto (71241075)
  Sinatrio Happy Triadji (71241093)

Usage:
    python app.py
"""

# Import required libraries for file handling and web server
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "agri_vision_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
IMG_SIZE = (224, 224)

# Default class ordering (overridden by class_names.txt if available)
DEFAULT_CLASS_NAMES = ["Brown_Spot", "Healthy", "Leaf_Blast", "Tungro"]

# ──────────────────────────────────────────────
# Organic treatment recommendations (Indonesian)
# Aligned with the project's SDG 2 (Zero Hunger) mission
# ──────────────────────────────────────────────
RECOMMENDATIONS = {
    "Healthy": (
        "Tanaman sehat! Pertahankan pemupukan organik teratur dan rotasi tanaman."
    ),
    "Brown_Spot": (
        "Gunakan fungisida nabati (ekstrak daun mimba atau cengkeh). "
        "Pastikan drainase air sawah baik dan gunakan pupuk kompos."
    ),
    "Leaf_Blast": (
        "Kurangi penggunaan pupuk nitrogen berlebih. "
        "Semprotkan agen hayati Trichoderma sp. atau ekstrak lengkuas."
    ),
    "Tungro": (
        "Cabut dan musnahkan tanaman yang terinfeksi. "
        "Kendalikan populasi wereng hijau (vektor virus) menggunakan "
        "pestisida nabati ekstrak sirsak atau bawang putih."
    ),
}

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ──────────────────────────────────────────────
# Model loading (lazy singleton)
# ──────────────────────────────────────────────
_model = None
_class_names = None


def get_model():
    """Load the Keras model once and cache it."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please run train.py first to generate the model."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
    return _model


def get_class_names():
    """Read class names saved during training, or fall back to defaults."""
    global _class_names
    if _class_names is None:
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                _class_names = [line.strip() for line in f if line.strip()]
            print(f"[INFO] Class names loaded: {_class_names}")
        else:
            _class_names = DEFAULT_CLASS_NAMES
            print(f"[WARN] class_names.txt not found — using defaults: {_class_names}")
    return _class_names


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image, resize it to 224x224, and apply MobileNetV2 preprocessing.
    This scales pixel values from [0, 255] to [-1, 1] to match training.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main Agri Vision UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept an uploaded rice leaf image, run inference, and return:
      - predicted_class
      - confidence_score (percentage string)
      - recommendation (organic treatment in Indonesian)
    """
    # ── Validate input ──────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong. Silakan pilih gambar."}), 400

    if not allowed_file(file.filename):
        return (
            jsonify({
                "error": (
                    "Format file tidak didukung. "
                    "Gunakan: PNG, JPG, JPEG, WEBP, atau BMP."
                )
            }),
            400,
        )

    # ── Save & preprocess ───────────────────────
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        img_array = preprocess_image(filepath)
    except Exception as e:
        return jsonify({"error": f"Gagal memproses gambar: {str(e)}"}), 400
    finally:
        # Clean up uploaded file after processing
        if os.path.exists(filepath):
            os.remove(filepath)

    # ── Inference ───────────────────────────────
    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    predictions = model.predict(img_array, verbose=0)
    class_names = get_class_names()

    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_idx]) * 100
    predicted_class = class_names[predicted_idx]

    recommendation = RECOMMENDATIONS.get(
        predicted_class,
        "Rekomendasi tidak tersedia untuk kelas ini.",
    )

    return jsonify({
        "predicted_class": predicted_class,
        "confidence_score": f"{confidence:.2f}%",
        "recommendation": recommendation,
    })


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  🌾 Agri Vision — Flask Server Starting")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
