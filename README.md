# FaceAttend AI

Automated face recognition attendance system using MTCNN + FaceNet + Flask.

No database required — all data stored in local files.

---

## Tech Stack

| Component       | Tool                  |
|-----------------|-----------------------|
| Face detection  | MTCNN                 |
| Face embedding  | FaceNet (keras-facenet)|
| Storage         | pickle + JSON         |
| Backend         | Flask                 |
| Frontend        | HTML + CSS + JS       |

---

## Project Structure

```
faceattend/
├── dataset/
│   ├── alice/          ← 20-50 images of Alice
│   └── bob/            ← 20-50 images of Bob
├── embeddings/
│   └── embeddings.pkl  ← auto-generated
├── attendance/
│   └── attendance.json ← auto-generated
├── templates/
│   └── index.html
├── train_embeddings.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect employee images

```
dataset/
  alice/
    photo1.jpg
    photo2.jpg
    ...
  bob/
    photo1.jpg
    ...
```

Take 20–50 photos per person. Include:
- Front face
- Slight left / right angles
- Different lighting conditions

### 3. Train embeddings (run once)

```bash
python train_embeddings.py
```

This creates `embeddings/embeddings.pkl`.

### 4. Start the server

```bash
python app.py
```

### 5. Open the dashboard

```
http://127.0.0.1:5000
```

Click **Start Recognition** — the browser webcam will scan for faces every 2 seconds
and automatically mark attendance.

---

## Local Files

**`embeddings/embeddings.pkl`**
```python
{
  "alice": [array([...128 floats...]), ...],
  "bob":   [array([...128 floats...]), ...]
}
```

**`attendance/attendance.json`**
```json
[
  { "name": "alice", "date": "2025-07-01", "time": "09:14:32" },
  { "name": "bob",   "date": "2025-07-01", "time": "09:22:10" }
]
```

---

## API Endpoints

| Method | Route               | Description                        |
|--------|---------------------|------------------------------------|
| GET    | `/`                 | Serves the HTML dashboard          |
| POST   | `/recognize`        | Accepts base64 image, returns name |
| GET    | `/attendance`       | Returns all attendance records     |
| GET    | `/attendance/today` | Returns today's records only       |
| POST   | `/clear`            | Clears all attendance records      |

---

## Tuning

Edit `THRESHOLD` in `app.py` (default `0.7`):
- Higher (e.g. `0.85`) → stricter matching, fewer false positives
- Lower  (e.g. `0.6`) → more lenient, better in bad lighting
