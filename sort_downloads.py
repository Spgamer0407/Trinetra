import json
import shutil
from pathlib import Path

from config import DATASET_DIR, ensure_dir

DOWNLOADS_DIR = Path("D:/sample vedios")
SOURCE_DIR = DOWNLOADS_DIR / "train_sample_videos"
METADATA_PATH = SOURCE_DIR / "metadata.json"
DEST_REAL = DATASET_DIR / "real"
DEST_FAKE = DATASET_DIR / "fake"


def sort_downloads(downloads_dir: str | Path = DOWNLOADS_DIR, target_count: int = 30):
    source_dir = Path(downloads_dir) / "train_sample_videos"
    metadata_path = source_dir / "metadata.json"

    if not metadata_path.exists():
        fallback_path = Path(downloads_dir) / "metadata.json"
        if fallback_path.exists():
            source_dir = Path(downloads_dir)
            metadata_path = fallback_path
        else:
            raise FileNotFoundError(f"Cannot find metadata.json at: {metadata_path}")

    ensure_dir(DEST_REAL)
    ensure_dir(DEST_FAKE)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    real_count = 0
    fake_count = 0

    print("Sorting balanced subset from Downloads...")
    for filename, info in metadata.items():
        src_file = source_dir / filename
        if not src_file.exists():
            continue

        label = str(info.get("label", "")).upper()

        if label == "REAL" and real_count < target_count:
            shutil.copy2(src_file, DEST_REAL / filename)
            real_count += 1
            print(f"[{real_count}/{target_count}] Copied REAL: {filename}")
        elif label == "FAKE" and fake_count < target_count:
            shutil.copy2(src_file, DEST_FAKE / filename)
            fake_count += 1
            print(f"[{fake_count}/{target_count}] Copied FAKE: {filename}")

        if real_count >= target_count and fake_count >= target_count:
            break

    print("\nDone sorting!")
    print(f"REAL: {real_count} videos copied to {DEST_REAL}")
    print(f"FAKE: {fake_count} videos copied to {DEST_FAKE}")
    return DEST_REAL, DEST_FAKE


def main():
    sort_downloads()


if __name__ == "__main__":
    main()