import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from config import (
    IMAGE_OUTPUT_DIR,
    MODEL_OUTPUT,
    REPORT_DIR,
    TARGET_SIZE,
    YUNET_MODEL_PATH,
    ensure_dir,
)

import cv2
import joblib
import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from explainability import generate_manipulation_explanation, get_gradcam_heatmap, overlay_heatmap
from forensic_fingerprint import (
    generate_digital_fingerprint,
    record_forensic_log,
)

MAX_DETECTION_DIMENSION = 1280


@dataclass(frozen=True)
class FaceCropResult:
    rank: int
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    label: str
    confidence: float
    fake_probability: float
    crop_path: Path
    gradcam_path: Optional[Path] = None
    explanation: Optional[str] = None


@dataclass(frozen=True)
class ImagePrediction:
    image_path: Path
    annotated_image_path: Optional[Path]
    verdict: str
    final_label: str
    confidence: Optional[float]
    fake_probability: Optional[float]
    real_probability: Optional[float]
    threshold: float
    faces: List[FaceCropResult]
    fingerprint: dict
    report_path: Path
    registered_in_ledger: bool
    forensic_explanation: Optional[str] = None


def load_yunet_detector(yunet_model_path: Path = YUNET_MODEL_PATH):
    if not yunet_model_path.is_file():
        raise FileNotFoundError(f"YuNet model not found: {yunet_model_path}")
    return cv2.FaceDetectorYN.create(
        model=str(yunet_model_path),
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=10,
    )


def detect_and_extract_faces(
    image: np.ndarray,
    detector,
    target_size=TARGET_SIZE,
    margin_ratio: float = 0.15,
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Detect faces in an image, returning a list of tuples: ((x, y, w, h), cropped_resized_face).
    """
    h, w = image.shape[:2]
    scale = min(1.0, MAX_DETECTION_DIMENSION / max(w, h))
    if scale < 1.0:
        detection_frame = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_frame = image

    detector.setInputSize((detection_frame.shape[1], detection_frame.shape[0]))
    _, faces = detector.detect(detection_frame)
    if faces is None or len(faces) == 0:
        return []

    # Sort faces by confidence descending
    sorted_faces = sorted(faces, key=lambda f: f[-1], reverse=True)[:5]
    results = []

    for face in sorted_faces:
        x, y, box_w, box_h = [int(v) for v in np.round(face[0:4] / scale)]

        margin_x = int(box_w * margin_ratio)
        margin_y = int(box_h * margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + box_w + margin_x)
        y2 = min(h, y + box_h + margin_y)

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        results.append(((x, y, box_w, box_h), resized))

    return results


def predict_face_embedding(crop: np.ndarray, feature_extractor, classifier) -> Tuple[str, float, float]:
    """
    Predict REAL vs FAKE for a 224x224 BGR face crop.
    Returns: (label, confidence, fake_probability)
    """
    rgb_face = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor_input = img_to_array(rgb_face)
    tensor_input = np.expand_dims(tensor_input, axis=0)
    tensor_input = preprocess_input(tensor_input)
    features = feature_extractor.predict(tensor_input, verbose=0)

    classes = np.asarray(classifier.classes_)
    fake_class_indices = np.flatnonzero(classes == 1)
    if len(fake_class_indices) != 1 or not np.array_equal(np.sort(classes), [0, 1]):
        raise ValueError(f"Classifier must have classes [0, 1]; got {classes.tolist()}.")

    probabilities = classifier.predict_proba(features)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = int(classes[predicted_index])
    fake_probability = float(probabilities[int(fake_class_indices[0])])
    label = "FAKE" if predicted_class == 1 else "REAL"
    confidence = float(probabilities[predicted_index])
    return label, confidence, fake_probability


def process_image(
    image_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] = IMAGE_OUTPUT_DIR,
    threshold: float = 0.5,
    explain: bool = True,
    register_ledger: bool = False,
    feature_extractor=None,
    classifier=None,
    gradcam_model=None,
    detector=None,
) -> ImagePrediction:
    """
    Analyze a single image for deepfake / manipulation artifacts:
    1. Computes digital forensic fingerprint (SHA-256 and pHash).
    2. Detects face(s) using YuNet.
    3. Classifies authenticity via DenseNet-201 + Random Forest.
    4. Computes Grad-CAM explainability heatmap overlay if explain=True.
    5. Saves annotated image with colored bounding boxes (Green=Real, Red=Fake).
    6. Optionally records authentic verified images into the forensic ledger.
    7. Emits a comprehensive forensic report JSON.
    """
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    out_dir = ensure_dir(output_dir)
    report_dir = ensure_dir(REPORT_DIR)

    # 1. Forensic Fingerprint
    fingerprint = generate_digital_fingerprint(image_path)
    print(f"\n[Trinetra] Loaded: {image_path.name}")
    print(f"  SHA-256: {fingerprint['sha256']}")
    print(f"  pHash:   {fingerprint['phash']}")

    # 2. Load Models if not passed in
    if detector is None:
        detector = load_yunet_detector()
    if feature_extractor is None:
        print("[Trinetra] Loading DenseNet-201 feature extractor...")
        feature_extractor = DenseNet201(weights="imagenet", include_top=False, pooling="avg")
        feature_extractor.trainable = False
    if classifier is None:
        if not MODEL_OUTPUT.is_file():
            raise FileNotFoundError(f"Classifier model not found at: {MODEL_OUTPUT}")
        classifier = joblib.load(MODEL_OUTPUT)
    if explain and gradcam_model is None:
        print("[Trinetra] Loading DenseNet-201 for Grad-CAM explainability...")
        gradcam_model = DenseNet201(weights="imagenet", include_top=True)

    # 3. Read image
    bgr_image = cv2.imread(str(image_path))
    if bgr_image is None:
        raise ValueError(f"Could not decode image at '{image_path}'. Ensure it is a valid image format.")

    annotated_image = bgr_image.copy()
    detected_faces = detect_and_extract_faces(bgr_image, detector)
    stem = image_path.stem

    face_results: List[FaceCropResult] = []
    fake_probs: List[float] = []

    if not detected_faces:
        print("  [Warning] No faces detected in image.")
        verdict = "NO FACE DETECTED"
        final_label = "UNCLASSIFIED"
        confidence = None
        fake_prob = None
        real_prob = None
        annotated_path = None
        forensic_explanation = "No human face detected for localization analysis."
    else:
        print(f"  Detected {len(detected_faces)} face(s). Running inference...")
        for rank, (box, crop) in enumerate(detected_faces, start=1):
            x, y, bw, bh = box
            label, conf, f_prob = predict_face_embedding(crop, feature_extractor, classifier)
            fake_probs.append(f_prob)

            # Save face crop
            crop_filename = out_dir / f"{stem}_face_rank{rank}.jpg"
            cv2.imwrite(str(crop_filename), crop)

            # Grad-CAM Explainability & Forensic Localization
            cam_path = None
            explanation = None
            if explain and gradcam_model is not None:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor_input = img_to_array(rgb_crop)
                tensor_input = np.expand_dims(tensor_input, axis=0)
                tensor_input = preprocess_input(tensor_input)

                heatmap = get_gradcam_heatmap(
                    tensor_input,
                    gradcam_model,
                    last_conv_layer_name="conv5_block32_concat",
                )
                _, superimposed = overlay_heatmap(crop, heatmap, alpha=0.4)
                cam_path = out_dir / f"{stem}_face_rank{rank}_gradcam.jpg"
                cv2.imwrite(str(cam_path), superimposed)

                # Synthesize 1-sentence localization explanation from Grad-CAM activation peaks
                explanation = generate_manipulation_explanation(
                    heatmap=heatmap,
                    is_fake=(label == "FAKE"),
                    fake_probability=f_prob,
                )
            else:
                if label == "FAKE":
                    explanation = f"Flagged as FAKE with {f_prob:.1%} manipulation probability (Grad-CAM visualization disabled)."
                else:
                    explanation = "Feature activations are naturally distributed with no localized synthetic manipulation anomalies."

            face_results.append(
                FaceCropResult(
                    rank=rank,
                    box=box,
                    label=label,
                    confidence=conf,
                    fake_probability=f_prob,
                    crop_path=crop_filename,
                    gradcam_path=cam_path,
                    explanation=explanation,
                )
            )

            # Annotate image
            # Red for FAKE, Green for REAL
            color = (0, 0, 255) if label == "FAKE" else (0, 200, 0)
            cv2.rectangle(annotated_image, (x, y), (x + bw, y + bh), color, 2)
            badge_text = f"{label} ({conf:.1%})"
            cv2.putText(
                annotated_image,
                badge_text,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            print(f"  Face #{rank}: {label} | Confidence: {conf:.1%} | Fake Prob: {f_prob:.1%}")
            if explanation:
                print(f"    -> Explanation: {explanation}")

        # Final verdict: if any face exceeds threshold, flag as fake
        max_fake_prob = max(fake_probs)
        fake_prob = max_fake_prob
        real_prob = 1.0 - max_fake_prob
        is_fake = max_fake_prob >= threshold
        verdict = "FAKE / MANIPULATED" if is_fake else "REAL / AUTHENTIC"
        final_label = "FAKE" if is_fake else "REAL"
        confidence = max_fake_prob if is_fake else (1.0 - max_fake_prob)

        # Primary forensic explanation for the overall image
        if is_fake:
            # Pick explanation of the highest fake probability face
            highest_fake_face = max(face_results, key=lambda f: f.fake_probability)
            forensic_explanation = highest_fake_face.explanation or "Potential manipulation artifacts flagged across facial regions."
        else:
            forensic_explanation = "All detected facial regions exhibit natural feature distributions consistent with authentic media."

        annotated_path = out_dir / f"{stem}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_image)

    # 4. Forensic Ledger Registration (for verified authentic content or when requested)
    registered = False
    if (final_label == "REAL" or register_ledger) and final_label != "UNCLASSIFIED":
        record_forensic_log(
            {
                "file_name": image_path.name,
                "timestamp": fingerprint["timestamp"],
                "sha256": fingerprint["sha256"],
                "phash": fingerprint["phash"],
                "verdict": final_label,
                "confidence": confidence,
            }
        )
        registered = True

    # 5. Save Forensic Report
    report_path = report_dir / f"{stem}_forensic_report.json"
    report_data = {
        "file_name": image_path.name,
        "file_path": str(image_path),
        "sha256": fingerprint["sha256"],
        "phash": fingerprint["phash"],
        "verdict": verdict,
        "final_label": final_label,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "threshold": threshold,
        "forensic_explanation": forensic_explanation,
        "faces_detected": len(face_results),
        "registered_in_ledger": registered,
        "faces": [
            {
                "rank": int(f.rank),
                "box": [int(v) for v in f.box],
                "label": str(f.label),
                "confidence": float(f.confidence),
                "fake_probability": float(f.fake_probability),
                "explanation": str(f.explanation) if f.explanation else None,
                "crop_path": str(f.crop_path),
                "gradcam_path": str(f.gradcam_path) if f.gradcam_path else None,
            }
            for f in face_results
        ],
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[Result] Verdict: {verdict}")
    if confidence is not None:
        print(f"         Confidence: {confidence:.1%}")
    print(f"         Forensic Explanation: {forensic_explanation}")
    print(f"         Forensic Report: {report_path}")
    if annotated_path:
        print(f"         Annotated Image: {annotated_path}")

    return ImagePrediction(
        image_path=image_path,
        annotated_image_path=annotated_path,
        verdict=verdict,
        final_label=final_label,
        confidence=confidence,
        fake_probability=fake_prob,
        real_probability=real_prob,
        threshold=threshold,
        faces=face_results,
        fingerprint=fingerprint,
        report_path=report_path,
        registered_in_ledger=registered,
        forensic_explanation=forensic_explanation,
    )


def main():
    parser = argparse.ArgumentParser(description="Trinetra: Deepfake Image Authenticity & Forensic Analysis")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fake probability threshold (default: 0.5)")
    parser.add_argument("--no-explain", dest="explain", action="store_false", help="Disable Grad-CAM heatmap generation")
    parser.add_argument("--register", action="store_true", help="Force register fingerprint in forensic ledger")
    args = parser.parse_args()

    process_image(
        image_path=args.image,
        threshold=args.threshold,
        explain=args.explain,
        register_ledger=args.register,
    )


if __name__ == "__main__":
    main()
