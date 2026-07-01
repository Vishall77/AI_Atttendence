import os
import pickle
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet

DATASET_PATH   = "dataset"
EMBEDDINGS_DIR = "embeddings"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.pkl")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

detector = MTCNN()
embedder = FaceNet()

print("=" * 50)
print("  FaceAttend AI — Training Embeddings")
print("=" * 50)


def detect_and_crop(img_rgb):
    """Run MTCNN on an RGB image, return cropped 160x160 face or None."""
    results = detector.detect_faces(img_rgb)
    if not results:
        return None
    # pick the highest-confidence detection
    best = max(results, key=lambda r: r["confidence"])
    x, y, w, h = best["box"]
    x, y = max(0, x), max(0, y)
    face = img_rgb[y : y + h, x : x + w]
    if face.size == 0:
        return None
    return cv2.resize(face, (160, 160))


def get_embedding(face_rgb):
    """Return 128-D FaceNet embedding for a 160x160 RGB face image."""
    face_f = face_rgb.astype("float32")
    face_f = np.expand_dims(face_f, axis=0)   # (1, 160, 160, 3)
    return embedder.embeddings(face_f)[0]       # (128,)


known_embeddings = {}   # { "alice": [emb1, emb2, ...], ... }

people = [
    p for p in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, p))
]

if not people:
    print(f"\nNo folders found inside '{DATASET_PATH}/'.")
    print("Create one folder per employee and put their photos inside.")
    exit(1)

for person_name in sorted(people):
    folder = os.path.join(DATASET_PATH, person_name)
    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    print(f"\nProcessing: {person_name} ({len(images)} images)")

    known_embeddings[person_name] = []

    for img_file in images:
        img_path = os.path.join(folder, img_file)
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [skip] Could not read {img_file}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        face    = detect_and_crop(img_rgb)

        if face is None:
            print(f"  [skip] No face detected in {img_file}")
            continue

        emb = get_embedding(face)
        known_embeddings[person_name].append(emb)
        print(f"  [ok]   {img_file}")

    count = len(known_embeddings[person_name])
    print(f"  => {count} embeddings saved for '{person_name}'")

# Save
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(known_embeddings, f)

print("\n" + "=" * 50)
print(f"  Done! Saved to {EMBEDDINGS_FILE}")
print(f"  Employees: {list(known_embeddings.keys())}")
print("=" * 50)
# print("\nNext step: python app.py")