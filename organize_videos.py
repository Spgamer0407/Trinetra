import json
import shutil
from pathlib import Path

from config import DATASET_DIR, ensure_dir

EXTRACTED_DIR = Path("D:/OPEN_CV/train_sample_videos")
METADATA_FILE = EXTRACTED_DIR / "metadata.json"

DEST_REAL = DATASET_DIR / "real"
DEST_FAKE = DATASET_DIR / "fake"


def organize_videos(extracted_dir: str | Path = EXTRACTED_DIR, limit_per_class: int = 30):
    extracted_dir = Path(extracted_dir)
    metadata_path = extracted_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot find metadata.json at: {metadata_path}")

    ensure_dir(DEST_REAL)
    ensure_dir(DEST_FAKE)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    real_copied = 0
    fake_copied = 0

    for filename, attributes in metadata.items():
        source_path = extracted_dir / filename
        if not source_path.exists():
            continue

        label = str(attributes.get("label", "")).upper()

        if label == "REAL" and real_copied < limit_per_class:
            shutil.copy2(source_path, DEST_REAL / filename)
            real_copied += 1
        elif label == "FAKE" and fake_copied < limit_per_class:
            shutil.copy2(source_path, DEST_FAKE / filename)
            fake_copied += 1

        if real_copied >= limit_per_class and fake_copied >= limit_per_class:
            break

    print(f"Successfully organized {real_copied} REAL and {fake_copied} FAKE videos.")
    return DEST_REAL, DEST_FAKE


def main():
    organize_videos()


if __name__ == "__main__":
    main()