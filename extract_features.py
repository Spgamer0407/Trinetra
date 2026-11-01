import os
from pathlib import Path

import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from config import FEATURES_PATH, LABELS_PATH, PROCESSED_DIR, TARGET_SIZE


CLASS_LABELS = {"real": 0, "fake": 1}
BATCH_SIZE = 16


def process_extracted_folder(input_dir: str | os.PathLike[str] = PROCESSED_DIR):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = []
    labels = []
    for class_name, label in CLASS_LABELS.items():
        class_dir = input_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Processed {class_name} directory not found: {class_dir}")

        class_images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not class_images:
            raise ValueError(f"No processed face images found in '{class_dir}'.")
        image_paths.extend(class_images)
        labels.extend([label] * len(class_images))

    print("Loading pre-trained DenseNet201 backbone...")
    feature_extractor = DenseNet201(weights="imagenet", include_top=False, pooling="avg")
    feature_extractor.trainable = False

    feature_batches = []
    print(f"Extracting features from {len(image_paths)} processed face crops...")
    for start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        batch = np.stack(
            [
                img_to_array(load_img(path, target_size=TARGET_SIZE))
                for path in batch_paths
            ]
        )
        batch = preprocess_input(batch)
        feature_batches.append(feature_extractor.predict(batch, verbose=0))

    features = np.concatenate(feature_batches, axis=0)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
    if features.shape != (len(image_paths), 1920):
        raise ValueError(
            f"Expected DenseNet201 features with shape ({len(image_paths)}, 1920), "
            f"got {features.shape}."
        )
    print(f"Feature extraction complete. Matrix shape: {features.shape}")
    print(f"Label matrix shape: {labels.shape}")
    return features, labels


def main():
    features, labels = process_extracted_folder(PROCESSED_DIR)
    np.save(FEATURES_PATH, features)
    np.save(LABELS_PATH, labels)
    print(f"Saved feature matrix to '{FEATURES_PATH}'.")
    print(f"Saved label matrix to '{LABELS_PATH}'.")


if __name__ == "__main__":
    main()