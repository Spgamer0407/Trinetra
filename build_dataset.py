import os
from pathlib import Path

import cv2

from config import DATASET_DIR, PROCESSED_DIR, TARGET_SIZE, YUNET_MODEL_PATH, ensure_dir


def crop_margin_face(frame, box, margin=0.15):
    h_img, w_img = frame.shape[:2]
    x, y, w, h = box
    dx, dy = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - dx), max(0, y - dy)
    x2, y2 = min(w_img, x + w + dx), min(h_img, y + h + dy)
    return frame[y1:y2, x1:x2]


def build_dataset(dataset_dir: str | os.PathLike[str] = DATASET_DIR, output_dir: str | os.PathLike[str] = PROCESSED_DIR, frames_per_video: int = 10):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    detector = cv2.FaceDetectorYN.create(
        model=str(YUNET_MODEL_PATH),
        config="",
        input_size=(320, 320),
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000,
    )

    for category in ["real", "fake"]:
        in_path = dataset_dir / category
        out_path = output_dir / category
        ensure_dir(out_path)

        if not in_path.exists():
            print(f"Skipping '{in_path}': directory does not exist.")
            continue

        videos = sorted([f for f in in_path.iterdir() if f.is_file() and f.suffix.lower() in {".mp4", ".avi", ".mov"}])
        print(f"\nProcessing {len(videos)} videos from '{category}'...")

        total_faces = 0
        for v_idx, v_name in enumerate(videos):
            cap = cv2.VideoCapture(str(v_name))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_count <= 0:
                cap.release()
                continue

            step = max(1, frame_count // frames_per_video)
            for f_idx in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                success, frame = cap.read()
                if not success or frame is None:
                    break

                h, w = frame.shape[:2]
                detector.setInputSize((w, h))
                _, faces = detector.detect(frame)

                if faces is not None and len(faces) > 0:
                    primary_face = max(faces, key=lambda f: f[2] * f[3])
                    box = primary_face[:4].astype(int)
                    crop = crop_margin_face(frame, box)

                    if crop.size > 0:
                        resized = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                        save_name = f"{category}_{v_idx:02d}_frame{f_idx:04d}.jpg"
                        cv2.imwrite(str(out_path / save_name), resized)
                        total_faces += 1

            cap.release()
        print(f"Finished '{category}': Saved {total_faces} face images.")

    return output_dir


def main():
    build_dataset()


if __name__ == "__main__":
    main()