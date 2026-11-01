import os
import urllib.request
from pathlib import Path

import cv2

from config import ROOT_DIR, TARGET_SIZE, YUNET_MODEL_PATH, ensure_dir

YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def ensure_model_exists():
    if not YUNET_MODEL_PATH.exists():
        print("Downloading YuNet face detection model (~330 KB)...")
        urllib.request.urlretrieve(YUNET_MODEL_URL, str(YUNET_MODEL_PATH))
        print("Download complete.")


def crop_and_resize_face(image_path: str | os.PathLike[str], output_path: str | os.PathLike[str], target_size=TARGET_SIZE) -> bool:
    ensure_model_exists()

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image at '{image_path}'. Ensure the file exists.")
        return False

    h, w, _ = image.shape
    detector = cv2.FaceDetectorYN.create(
        model=str(YUNET_MODEL_PATH),
        config="",
        input_size=(w, h),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=10,
    )

    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        print("No face detected in the image.")
        return False

    best_face = max(faces, key=lambda f: f[-1])
    x, y, box_w, box_h = best_face[0:4].astype(int)

    margin_x = int(box_w * 0.15)
    margin_y = int(box_h * 0.15)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + box_w + margin_x)
    y2 = min(h, y + box_h + margin_y)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        print("Error: Empty crop region.")
        return False

    resized_face = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), resized_face)
    print(f"Success! Cropped face saved to '{output_path}' with dimensions {resized_face.shape}.")
    return True


def main():
    crop_and_resize_face(ROOT_DIR / "test_face.jpg", ROOT_DIR / "cropped_224.jpg")


if __name__ == "__main__":
    main()