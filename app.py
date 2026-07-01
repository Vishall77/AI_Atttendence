import os
import json
import pickle
import base64
import numpy as np
import cv2
from datetime import datetime, date
from flask import Flask, render_template, request, jsonify
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = "embeddings/embeddings.pkl"
ATTENDANCE_FILE = "attendance/attendance.json"
THRESHOLD       = 0.5          # cosine similarity cutoff

os.makedirs("attendance", exist_ok=True)

# ── App init ───────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load models ────────────────────────────────────────────────────────────────
print("[FaceAttend] Loading models...")
detector = MTCNN()
embedder = FaceNet()
print("[FaceAttend] Models ready.")

# ── Load embeddings ────────────────────────────────────────────────────────────
if not os.path.exists(EMBEDDINGS_FILE):
    raise FileNotFoundError(
        f"'{EMBEDDINGS_FILE}' not found. Run train_embeddings.py first."
    )

with open(EMBEDDINGS_FILE, "rb") as f:
    known_embeddings = pickle.load(f)

# Flatten into two parallel lists for matrix comparison
names_flat      = []
embeddings_flat = []
for name, embs in known_embeddings.items():
    for emb in embs:
        names_flat.append(name)
        embeddings_flat.append(emb)

embeddings_matrix = np.array(embeddings_flat)   # shape (N, 128)
print(f"[FaceAttend] Loaded {len(set(names_flat))} employee(s), "
      f"{len(names_flat)} embedding(s).")


# ── Attendance helpers ──────────────────────────────────────────────────────────
def load_records():
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    with open(ATTENDANCE_FILE, "r") as f:
        return json.load(f)


def save_records(records):
    with open(ATTENDANCE_FILE, "w") as f:
        json.dump(records, f, indent=2)


def already_marked(name: str, today: str) -> bool:
    return any(
        r["name"] == name and r["date"] == today
        for r in load_records()
    )


def mark_attendance(name: str) -> bool:
    today    = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M:%S")

    if already_marked(name, today):
        return False   # duplicate — not saved again

    records = load_records()
    records.append({"name": name, "date": today, "time": now_time})
    save_records(records)
    return True


# ── Face recognition helper ─────────────────────────────────────────────────────
def recognize(face_rgb: np.ndarray):
    """
    face_rgb: (160, 160, 3) uint8 RGB image
    Returns: (name | None, score)
    """
    face_f    = face_rgb.astype("float32")
    embedding = embedder.embeddings(np.expand_dims(face_f, 0))[0]  # (128,)
    scores    = cosine_similarity([embedding], embeddings_matrix)[0]
    best_idx  = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= THRESHOLD:
        return names_flat[best_idx], best_score
    return None, best_score


# ── Routes ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize_route():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"status": "error", "message": "No image provided"}), 400

    # Decode base64 frame from the browser
    try:
        header, encoded = data["image"].split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    if frame_bgr is None:
        return jsonify({"status": "error", "message": "Could not decode image"}), 400

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Detect face
    results = detector.detect_faces(frame_rgb)
    if not results:
        return jsonify({"status": "no_face"})

    # Crop best face
    best = max(results, key=lambda r: r["confidence"])
    x, y, w, h = best["box"]
    x, y = max(0, x), max(0, y)
    face = frame_rgb[y : y + h, x : x + w]
    if face.size == 0:
        return jsonify({"status": "no_face"})
    face = cv2.resize(face, (160, 160))

    # Recognize
    name, score = recognize(face)

    if name:
        marked = mark_attendance(name)
        return jsonify({
            "status": "recognized",
            "name":   name,
            "score":  round(score, 4),
            "marked": marked,
        })

    return jsonify({
        "status": "We don't Recognize you",
        "score":  round(score, 4),
    })


@app.route("/attendance")
def attendance_all():
    records = load_records()
    return jsonify(records)


@app.route("/attendance/today")
def attendance_today():
    today   = date.today().isoformat()
    records = [r for r in load_records() if r["date"] == today]
    return jsonify(records)


@app.route("/clear", methods=["POST"])
def clear_attendance():
    save_records([])
    return jsonify({"status": "ok", "message": "Attendance cleared."})


# ── Run ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[FaceAttend] Server running → http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)