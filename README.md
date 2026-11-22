# TRINETRA: AI-Assisted Digital Authenticity Verification and Forensic Traceability System

TRINETRA is a unified framework combining deepfake face manipulation detection, visual explainability (Grad-CAM), and dual-layer cryptographic and perceptual forensic traceability (SHA-256 and pHash).

Developed for **Visvesvaraya Technological University (VTU)** mini-project curriculum at **Sahyadri College of Engineering & Management, Mangaluru (2025-26)**.

---

## Key Capabilities

1. **Single Image Fake Detection (`process-image`)**:
   - Detects and extracts facial regions with dynamic boundary padding using OpenCV YuNet.
   - Extracts 1,920-dimensional deep feature embeddings using **DenseNet-201**.
   - Classifies authenticity (REAL vs. FAKE) using a trained Random Forest model.
   - Generates annotated images with colored bounding boxes (Green: REAL, Red: FAKE).
   - Generates visual explainability heatmaps using **Grad-CAM**.
   - Emits dual-layer forensic fingerprints and structured JSON reports.

2. **Video Deepfake Analysis (`process-video`)**:
   - Samples frames across input videos, detects faces, aggregates per-frame probabilities, and saves frame-level pHashes for flagged fake frames.

3. **Forensic Integrity Verification (`verify-integrity`)**:
   - Compares candidate images/videos against the immutable forensic ledger (`forensic_ledger.json`).
   - Distinguishes between:
     - **Exact Match (SHA-256)**: 100% bit-level unaltered authenticity.
     - **Perceptual Match (pHash Hamming distance $\le 5$)**: Authentic content with compression, format, or resolution changes.
     - **Suspicious Similarity (distance 6-10)**: Content sharing visual structure but showing localized alterations.
     - **Untracked / Not Found**: Content not present in the forensic ledger.

4. **Interactive Web Dashboard (`web`)**:
   - Clean, zero-dependency browser interface running on `http://127.0.0.1:8080`.
   - Drag-and-drop image analysis, real-time Grad-CAM viewing, one-click ledger verification, and ledger browsing.

---

## Project Structure

- `config.py` – Global paths, environment settings (`KERAS_HOME`), and thresholds
- `main.py` – Unified command-line entry point
- `web_app.py` – Interactive web application dashboard
- `process_image.py` – Single image fake detection and annotation pipeline
- `process_video.py` – Video frame sampling, face detection, and batch classification
- `forensic_fingerprint.py` – SHA-256, perceptual hash (pHash), ledger logging, and integrity verification
- `explainability.py` – Grad-CAM heatmap generation and overlay helpers
- `build_dataset.py` – Extracts face crops from dataset videos into `processed_faces/`
- `extract_features.py` – Extracts DenseNet-201 embeddings into `features.npy` and `labels.npy`
- `train_classifier.py` – Trains Random Forest model on video-level held-out splits
- `metrics_report.json` – Test evaluation metrics (accuracy: 83.47%, precision, recall, F1)
- `forensic_ledger.json` – Persistent ledger of authentic fingerprints

---

## Quickstart & Usage

### 1. Launch the Interactive Web Dashboard
Run:
```bash
python main.py web
```
Then open your browser at **`http://127.0.0.1:8080`**.

### 2. Single Image Analysis (CLI)
Analyze an image for manipulation and generate Grad-CAM visual evidence:
```bash
python main.py process-image test_face.jpg
```
Options:
- `--threshold 0.5`: Adjust fake probability sensitivity (default: 0.5)
- `--no-explain`: Disable Grad-CAM heatmap generation
- `--register`: Force register fingerprint into the forensic ledger

Output files:
- Annotated Image: `outputs/images/<name>_annotated.jpg`
- Face Crop: `outputs/images/<name>_face_rank1.jpg`
- Grad-CAM Heatmap: `outputs/images/<name>_face_rank1_gradcam.jpg`
- Forensic Report: `outputs/reports/<name>_forensic_report.json`

### 3. Verify Content Integrity against Ledger
Verify if a file has been registered or altered:
```bash
python main.py verify-integrity test_face.jpg
```

### 4. Video Analysis (CLI)
Classify a video across sampled frames:
```bash
python main.py process-video sample.mp4 --frames 15 --threshold 0.5 --explain
```

### 5. Model Training & Dataset Pipeline
To retrain or rebuild features from scratch:
```bash
python main.py build-dataset
python main.py extract-features
python main.py train
```

---

## Evaluation Performance (`metrics_report.json`)

Trained and evaluated on video-level held-out splits (preventing frame-to-frame data leakage):
- **Accuracy:** `83.47%`
- **REAL Class:** Precision: `100%`, Recall: `66.67%`, F1-score: `0.80`
- **FAKE Class:** Precision: `75.31%`, Recall: `100%`, F1-score: `0.86`

---

## Explainability Context (Viva Note)

DenseNet-201 acts as the spatial feature extractor whose pooled representation is classified by the Random Forest. Grad-CAM heatmaps highlight feature activation regions at the final dense block (`conv5_block32_concat`), providing visual evidence of salient facial features.
