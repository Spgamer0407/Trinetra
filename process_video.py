import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from config import (
    DEFAULT_NUM_FRAMES,
    MODEL_OUTPUT,
    OUTPUT_DIR,
    REPORT_DIR,
    SAMPLE_VIDEO,
    TARGET_SIZE,
    YUNET_MODEL_PATH,
    ensure_dir,
)
from explainability import get_gradcam_heatmap, overlay_heatmap
from forensic_fingerprint import calculate_phash, calculate_sha256

MAX_DETECTION_DIMENSION = 1280


@dataclass(frozen=True)
class FacePrediction:
    frame_index: int
    face_rank: int
    label: str
    confidence: float
    fake_probability: float
    crop_path: Path


@dataclass(frozen=True)
class VideoPrediction:
    video_path: Path
    verdict: str
    threshold: float
    mean_fake_probability: float | None
    frames_analyzed: int
    face_predictions: list[FacePrediction]
    output_dir: Path
    report_path: Path


def get_face_crop(frame: np.ndarray, detector, target_size=TARGET_SIZE):
    h, w, _ = frame.shape
    scale = min(1.0, MAX_DETECTION_DIMENSION / max(w, h))
    if scale < 1.0:
        detection_frame = cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_frame = frame

    detector.setInputSize((detection_frame.shape[1], detection_frame.shape[0]))

    _, faces = detector.detect(detection_frame)
    if faces is None or len(faces) == 0:
        return []

    sorted_faces = sorted(faces, key=lambda f: f[-1], reverse=True)[:3]
    crops = []

    for face in sorted_faces:
        x, y, box_w, box_h = np.round(face[0:4] / scale).astype(int)

        margin_x = int(box_w * 0.15)
        margin_y = int(box_h * 0.15)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + box_w + margin_x)
        y2 = min(h, y + box_h + margin_y)

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        crops.append(cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA))

    return crops


def predict_face(crop: np.ndarray, feature_extractor, classifier) -> tuple[str, float, float]:
    rgb_face = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor_input = img_to_array(rgb_face)
    tensor_input = np.expand_dims(tensor_input, axis=0)
    tensor_input = preprocess_input(tensor_input)
    features = feature_extractor.predict(tensor_input, verbose=0)

    classes = np.asarray(classifier.classes_)
    fake_class_indices = np.flatnonzero(classes == 1)
    if len(fake_class_indices) != 1 or not np.array_equal(np.sort(classes), [0, 1]):
        raise ValueError(
            "The classifier must have classes 0 (REAL) and 1 (FAKE); "
            f"found {classes.tolist()}."
        )

    probabilities = classifier.predict_proba(features)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = int(classes[predicted_index])
    fake_probability = float(probabilities[int(fake_class_indices[0])])
    label = "FAKE" if predicted_class == 1 else "REAL"
    confidence = float(probabilities[predicted_index])
    return label, confidence, fake_probability


def process_video(
    video_path: str | os.PathLike[str] = SAMPLE_VIDEO,
    output_dir: str | os.PathLike[str] = OUTPUT_DIR,
    num_frames: int = DEFAULT_NUM_FRAMES,
    explain: bool = False,
    threshold: float = 0.5,
) -> VideoPrediction:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than zero.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")

    ensure_dir(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    sha256 = calculate_sha256(video_path)
    if not YUNET_MODEL_PATH.is_file():
        raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL_PATH}")
    if not MODEL_OUTPUT.is_file():
        raise FileNotFoundError(f"Random Forest model not found: {MODEL_OUTPUT}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Loaded '{video_path}': {total_frames} total frames at {fps:.2f} FPS.")

        if total_frames < num_frames:
            frame_indices = np.arange(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        detector = cv2.FaceDetectorYN.create(
            model=str(YUNET_MODEL_PATH),
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5,
        )

        print("[Prediction] Loading DenseNet201 feature extractor and Random Forest classifier...")
        feature_extractor = DenseNet201(
            weights="imagenet",
            include_top=False,
            pooling="avg",
        )
        feature_extractor.trainable = False
        classifier = joblib.load(MODEL_OUTPUT)

        gradcam_model = None
        if explain:
            print("[Explainability] Loading DenseNet201 for Grad-CAM...")
            gradcam_model = DenseNet201(weights="imagenet", include_top=True)

        face_predictions = []
        face_fake_probabilities = []
        frames_analyzed = 0
        flagged_frames = []
        flagged_frame_phashes = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            success, frame = cap.read()
            if not success or frame is None:
                continue

            cropped_faces = get_face_crop(frame, detector)
            if not cropped_faces:
                continue

            frame_is_fake = False
            for rank, cropped_face in enumerate(cropped_faces, start=1):
                filename = output_dir / f"face_frame_{idx:05d}_rank_{rank}.jpg"
                cv2.imwrite(str(filename), cropped_face)
                label, confidence, fake_probability = predict_face(
                    cropped_face,
                    feature_extractor,
                    classifier,
                )
                face_predictions.append(
                    FacePrediction(
                        frame_index=int(idx),
                        face_rank=rank,
                        label=label,
                        confidence=confidence,
                        fake_probability=fake_probability,
                        crop_path=filename,
                    )
                )
                face_fake_probabilities.append(fake_probability)
                frame_is_fake = frame_is_fake or label == "FAKE"
                print(
                    f"  Frame {idx}, face {rank}: {label} "
                    f"(confidence {confidence:.1%}, fake probability {fake_probability:.1%})"
                )

                if explain and gradcam_model is not None:
                    rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    tensor_input = img_to_array(rgb_face)
                    tensor_input = np.expand_dims(tensor_input, axis=0)
                    tensor_input = preprocess_input(tensor_input)

                    heatmap = get_gradcam_heatmap(
                        tensor_input,
                        gradcam_model,
                        last_conv_layer_name="conv5_block32_concat",
                    )

                    _, superimposed = overlay_heatmap(cropped_face, heatmap, alpha=0.4)
                    out_cam_path = output_dir / f"{video_path.stem}_frame_{idx:05d}_rank_{rank}_gradcam.jpg"
                    cv2.imwrite(str(out_cam_path), superimposed)
                    print(f"  [Explainability] Saved visual evidence to: {out_cam_path}")

            frames_analyzed += 1
            if frame_is_fake:
                frame_index = int(idx)
                flagged_frames.append(frame_index)
                frame_phash = calculate_phash(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
                flagged_frame_phashes.append(
                    {"frame_index": frame_index, "phash": frame_phash}
                )
    finally:
        cap.release()

    mean_fake_probability = (
        float(np.mean(face_fake_probabilities)) if face_fake_probabilities else None
    )
    if mean_fake_probability is None:
        verdict = "NO FACE DETECTED"
        final_verdict = "UNCLASSIFIED"
        confidence = None
        print("No faces were detected in the sampled video frames; unable to classify video.")
    else:
        is_fake = mean_fake_probability >= threshold
        verdict = "FAKE / MANIPULATED" if is_fake else "REAL / AUTHENTIC"
        final_verdict = "FAKE" if is_fake else "REAL"
        confidence = mean_fake_probability if is_fake else 1.0 - mean_fake_probability
        print(
            f"Video verdict: {verdict} "
            f"(mean fake probability {mean_fake_probability:.1%} across "
            f"{frames_analyzed} analyzed frames; threshold {threshold:.1%})."
        )

    report_dir = ensure_dir(REPORT_DIR)
    report_path = report_dir / f"{video_path.stem}_forensic_report.json"
    report = {
        "file_name": video_path.name,
        "sha256": sha256,
        "final_verdict": final_verdict,
        "confidence": confidence,
        "total_frames_analyzed": frames_analyzed,
        "flagged_frames": flagged_frames,
        "flagged_frame_phashes": flagged_frame_phashes,
    }
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    print(f"Saved forensic report to '{report_path}'.")
    print(f"Saved {len(face_predictions)} face crops to '{output_dir}'.")
    return VideoPrediction(
        video_path=video_path,
        verdict=verdict,
        threshold=threshold,
        mean_fake_probability=mean_fake_probability,
        frames_analyzed=frames_analyzed,
        face_predictions=face_predictions,
        output_dir=output_dir,
        report_path=report_path,
    )


def main(
    explain: bool = False,
    video_path: str | os.PathLike[str] | None = None,
    num_frames: int = DEFAULT_NUM_FRAMES,
    threshold: float = 0.5,
) -> VideoPrediction:
    return process_video(
        video_path=SAMPLE_VIDEO if video_path is None else video_path,
        num_frames=num_frames,
        explain=explain,
        threshold=threshold,
    )


if __name__ == "__main__":
    main()