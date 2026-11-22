import os
import config
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Import your corrected explainability functions
from explainability import get_gradcam_heatmap, overlay_heatmap

# 1. Target a face crop from processed_faces (aligned with training data)
sample_dir = r"D:\OPEN_CV\processed_faces\real"
if not os.path.exists(sample_dir) or len(os.listdir(sample_dir)) == 0:
    sample_dir = r"D:\OPEN_CV\processed_faces\fake"

sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png'))]
if not sample_files:
    raise FileNotFoundError("No processed face crops found in processed_faces to test.")

test_img_path = os.path.join(sample_dir, sample_files[0])
print(f"[1/4] Selected test face crop: {test_img_path}")

# 2. Read and prepare input image
bgr_img = cv2.imread(test_img_path)
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_AREA)

tensor_input = img_to_array(resized)
tensor_input = np.expand_dims(tensor_input, axis=0)
tensor_input = preprocess_input(tensor_input)

# 3. Load base DenseNet-201 with top layer enabled for gradient backpropagation
print("[2/4] Initializing DenseNet-201 model graph...")
model = DenseNet201(weights="imagenet", include_top=True)

# 4. Compute Grad-CAM heatmap
print("[3/4] Computing Grad-CAM heatmap at 'conv5_block32_concat'...")
heatmap = get_gradcam_heatmap(tensor_input, model, last_conv_layer_name="conv5_block32_concat")

# Shape & Value checks
print(f"      Heatmap shape: {heatmap.shape} (min: {heatmap.min():.3f}, max: {heatmap.max():.3f})")
assert heatmap.ndim == 2, f"Heatmap should be 2D, got shape {heatmap.shape}"
assert 0.0 <= np.min(heatmap) and np.max(heatmap) <= 1.0, "Heatmap values not scaled in [0, 1]"

# 5. Overlay heatmap onto source face image
print("[4/4] Blending heatmap overlay...")
colored_cam, superimposed = overlay_heatmap(bgr_img, heatmap, alpha=0.4)

out_file = r"D:\OPEN_CV\gradcam_validation_output.jpg"
cv2.imwrite(out_file, superimposed)
print(f"\nSUCCESS: Validation passed! Output saved to: {out_file}")