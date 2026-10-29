import os
import glob
import numpy as np
from keras.applications.densenet import DenseNet201, preprocess_input
from keras.utils import load_img, img_to_array

# 1. Load DenseNet201 with ImageNet weights, excluding top classification layer
print("Loading pre-trained DenseNet201 backbone...")
base_model = DenseNet201(weights="imagenet", include_top=False, pooling="avg")
base_model.trainable = False  # Freeze weights (Fixed backbone)

def extract_vector(image_path: str) -> np.ndarray:
    """Loads a 224x224 image and returns a 1920-dimensional feature vector."""
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features = base_model.predict(x, verbose=0)
    return features.flatten()

def process_extracted_folder(input_dir: str):
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    if not image_paths:
        print(f"No .jpg frames found in '{input_dir}'.")
        return None

    feature_matrix = []
    print(f"Extracting features from {len(image_paths)} cropped faces...")
    
    for path in image_paths:
        vec = extract_vector(path)
        feature_matrix.append(vec)

    feature_matrix = np.array(feature_matrix)
    print(f"Feature extraction complete. Matrix shape: {feature_matrix.shape}")
    return feature_matrix

if __name__ == "__main__":
    features = process_extracted_folder("output_faces")
    if features is not None:
        np.save("extracted_features.npy", features)
        print("Saved extracted feature matrix to 'extracted_features.npy'.")