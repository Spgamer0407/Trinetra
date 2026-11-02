import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import (
    FEATURES_PATH,
    LABELS_PATH,
    METRICS_REPORT_PATH,
    MODEL_OUTPUT,
    PROCESSED_DIR,
)

VIDEO_FRAME_PATTERN = re.compile(r"(real|fake)_(\d+)_frame\d+\.(?:jpg|png)", re.IGNORECASE)
TEST_SIZE = 0.2


def load_dataset(processed_dir: str | os.PathLike[str] = PROCESSED_DIR):
    processed_dir = Path(processed_dir)
    classes = {"real": 0, "fake": 1}
    samples = []
    video_labels = {}

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {processed_dir}")

    for label_name, label_id in classes.items():
        folder = processed_dir / label_name
        if not folder.exists():
            continue

        images = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".png"}])
        for img_file in images:
            match = VIDEO_FRAME_PATTERN.fullmatch(img_file.name)
            if match is None or match.group(1).lower() != label_name:
                raise ValueError(
                    f"Cannot determine the source video for '{img_file.name}'. "
                    "Expected filenames like 'real_00_frame0000.jpg' or 'fake_00_frame0000.jpg'."
                )

            video_id = f"{label_name}_{match.group(2)}"
            video_labels[video_id] = label_id
            samples.append((str(img_file), label_id, video_id))

    if not samples:
        raise ValueError(f"No processed face images found in '{processed_dir}'.")

    return samples, video_labels


def train_classifier(processed_dir: str | os.PathLike[str] = PROCESSED_DIR, model_output: str | os.PathLike[str] = MODEL_OUTPUT):
    samples, video_labels = load_dataset(processed_dir)
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH).reshape(-1)

    expected_labels = np.asarray([sample[1] for sample in samples])
    if features.shape != (len(samples), 1920):
        raise ValueError(
            f"Feature cache must have shape ({len(samples)}, 1920), got {features.shape}. "
            "Run 'python main.py extract-features' to rebuild it."
        )
    if labels.shape != expected_labels.shape or not np.array_equal(labels, expected_labels):
        raise ValueError(
            "Label cache does not match the processed face images. "
            "Run 'python main.py extract-features' to rebuild it."
        )

    video_ids = np.array(list(video_labels))
    video_targets = np.array([video_labels[video_id] for video_id in video_ids])

    train_videos, test_videos = train_test_split(
        video_ids,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=video_targets,
    )
    train_video_set = set(train_videos)
    test_video_set = set(test_videos)

    train_indices = np.asarray(
        [index for index, sample in enumerate(samples) if sample[2] in train_video_set]
    )
    test_indices = np.asarray(
        [index for index, sample in enumerate(samples) if sample[2] in test_video_set]
    )

    print(f"Video-level split: {len(train_video_set)} training videos, {len(test_video_set)} test videos.")
    print(f"Using cached features for {len(train_indices)} training faces and {len(test_indices)} test faces.")
    X_train, y_train = features[train_indices], labels[train_indices]
    X_test, y_test = features[test_indices], labels[test_indices]

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, preds))
    report = classification_report(
        y_test,
        preds,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, preds, labels=[0, 1])
    true_real, false_fake, false_real, true_fake = matrix.ravel()
    evaluation = {
        "evaluation_unit": "face crops from held-out videos",
        "video_level_split": {
            "training_videos": len(train_video_set),
            "test_videos": len(test_video_set),
            "test_faces": len(test_indices),
        },
        "accuracy": accuracy,
        "confusion_matrix": {
            "labels": ["REAL", "FAKE"],
            "rows": "actual",
            "columns": "predicted",
            "matrix": matrix.tolist(),
        },
        "confusion_matrix_counts": {
            "true_real": int(true_real),
            "false_real": int(false_real),
            "true_fake": int(true_fake),
            "false_fake": int(false_fake),
        },
        "classification_report": report,
    }
    with METRICS_REPORT_PATH.open("w", encoding="utf-8") as report_file:
        json.dump(evaluation, report_file, indent=2)

    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(
        y_test,
        preds,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        zero_division=0,
    ))
    print(f"Evaluation metrics saved to: {METRICS_REPORT_PATH}")

    joblib.dump(clf, model_output)
    print(f"Trained model saved successfully to: {model_output}")
    return clf


def main():
    train_classifier()


if __name__ == "__main__":
    main()