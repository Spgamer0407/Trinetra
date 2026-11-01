from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATASET_DIR = ROOT_DIR / "dataset"
PROCESSED_DIR = ROOT_DIR / "processed_faces"
OUTPUT_DIR = ROOT_DIR / "output_faces"
REPORT_DIR = ROOT_DIR / "outputs" / "reports"
YUNET_MODEL_PATH = ROOT_DIR / "face_detection_yunet_2023mar.onnx"
MODEL_OUTPUT = ROOT_DIR / "trinetra_rf_model.pkl"
FEATURES_PATH = ROOT_DIR / "features.npy"
LABELS_PATH = ROOT_DIR / "labels.npy"
SAMPLE_VIDEO = ROOT_DIR / "sample.mp4"
FORENSIC_LEDGER = ROOT_DIR / "forensic_ledger.json"

FRAMES_PER_VIDEO = 10
TARGET_SIZE = (224, 224)
MARGIN = 0.15
DEFAULT_NUM_FRAMES = 15


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
