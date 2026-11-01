import cv2
import numpy as np
import tensorflow as tf


def get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block32_concat"):
    """
    Computes a Grad-CAM heatmap for an input image tensor using DenseNet201.

    Args:
        img_array: 4D numpy array or tensor with shape (1, 224, 224, 3), preprocessed.
        model: Loaded Keras model instance with weights.
        last_conv_layer_name: Name of the target feature layer (default: final dense block concat).

    Returns:
        2D numpy array representing the normalized heatmap in range [0, 1].
    """
    # 1. Isolate the target feature map layer and model predictions
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    # 2. Record forward pass and calculate gradients
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model([img_array])

        # Ensure preds has 2D shape (batch_size, num_classes)
        # Squeeze in case GlobalAveragePooling wasn't applied or dims are redundant
        if tf.rank(preds) > 2:
            preds = tf.squeeze(preds)
            if tf.rank(preds) == 1:
                preds = tf.expand_dims(preds, axis=0)

        # Get target class index and explicit score channel
        predicted_index = tf.argmax(preds[0])
        class_channel = preds[0, predicted_index]

    # Gradients of target class score with respect to feature maps (shape: [1, H, W, C])
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError(
            f"Could not compute gradients for layer '{last_conv_layer_name}'. "
            "Ensure the layer is part of the differentiable forward path."
        )

    # 3. Global average pooling over spatial dimensions -> shape: (1, C)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # 4. Proper 4D broadcasting:
    # conv_outputs: (1, H, W, C)
    # pooled_grads[:, tf.newaxis, tf.newaxis, :]: (1, 1, 1, C)
    cam = tf.reduce_sum(
        conv_outputs * pooled_grads[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )  # Resulting shape: (1, H, W)

    # 5. Remove batch dimension -> shape: (H, W)
    cam = cam[0]

    # 6. Apply ReLU to isolate features that positively influence the target class
    cam = tf.maximum(cam, 0.0)

    # 7. Normalize to [0, 1] avoiding division by zero
    max_val = tf.math.reduce_max(cam)
    if max_val != 0:
        cam = cam / max_val

    return cam.numpy()


def overlay_heatmap(image_bgr, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Superimposes a Grad-CAM heatmap over the original image.

    Args:
        image_bgr: Source OpenCV image (H, W, 3) in uint8 format.
        heatmap: 2D float array in [0, 1].
        alpha: Blending weight for heatmap overlay (0.0 to 1.0).

    Returns:
        Tuple: (colored_heatmap, superimposed_bgr)
    """
    h, w = image_bgr.shape[:2]

    # Resize heatmap to match original image dimensions
    resized_heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * resized_heatmap)

    # Convert single-channel heatmap to RGB/BGR false-color representation
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

    # Alpha-blend the heatmap with original image
    superimposed = cv2.addWeighted(colored_heatmap, alpha, image_bgr, 1.0 - alpha, 0)

    return colored_heatmap, superimposed