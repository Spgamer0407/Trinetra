# Trinetra

A lightweight deepfake detection pipeline for face extraction, feature embedding, and training a Random Forest classifier.

## Structure

- `config.py` – shared project paths and defaults
- `main.py` – top-level CLI entry point
- `process_video.py` – extracts face crops from a video
- `build_dataset.py` – builds processed face images from dataset folders
- `extract_features.py` – exports DenseNet embeddings and labels from `processed_faces/`
- `train_classifier.py` – trains the Random Forest model
- `explainability.py` – Grad-CAM style visual explanation helpers
- `crop_face.py` – single-image face crop utility
- `forensic_fingerprint.py` – forensic hash generation and ledger logging

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the YuNet ONNX model in the project root as:

   `face_detection_yunet_2023mar.onnx`

4. Add your dataset folders:

   ```text
   dataset/
     real/
     fake/
   ```

## Typical workflow

```bash
python main.py process-video [video.mp4] [--frames 15] [--threshold 0.5]
python main.py build-dataset
python main.py extract-features
python main.py train
```

`extract-features` reads all face crops from `processed_faces/real` and
`processed_faces/fake`, then saves the 1,920-dimensional embeddings to
`features.npy` and their labels (`0` for REAL, `1` for FAKE) to `labels.npy`.
`train` uses these cached arrays while retaining a video-level train/test split.
Re-run `extract-features` after changing the processed crops.

`process-video` samples the requested number of frames, detects faces, and classifies each face with the trained DenseNet-201 and Random Forest pipeline. It reports the mean fake probability across analyzed frames; values above the threshold are labeled **FAKE / MANIPULATED**, otherwise **REAL / AUTHENTIC**. If no faces are detected, it reports that the video could not be classified. Face crops are saved in `output_faces/`.
Each run also saves a forensic report to `outputs/reports/<video_name>_forensic_report.json`. The report includes the input file's SHA-256, the verdict and confidence, the number of frames analyzed, and frame indices with at least one FAKE face prediction alongside pHashes for those full video frames. Confidence is the estimated probability of the reported REAL or FAKE verdict; videos with no classifiable faces have an `UNCLASSIFIED` verdict and a null confidence.

If you want to run a direct script instead of the CLI:

```bash
python process_video.py
python build_dataset.py
python extract_features.py
python train_classifier.py
```

## Notes

- The project is designed to run from the repository root.
- It uses relative paths via `config.py` rather than hardcoded Windows drive paths.
- You may need a CUDA-capable TensorFlow setup for large-scale training, but CPU mode is supported.
