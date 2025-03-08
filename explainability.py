import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model


def compute_gradcam(model, img_array, layer_name, class_idx=0):
    """
    Compute Grad-CAM visualization for model interpretability.

    Args:
        model: Trained model
        img_array: Input image array
        layer_name: Name of the layer to use for Grad-CAM
        class_idx: Index of the class to explain (0 for real, 1 for fake)

    Returns:
        Heatmap array
    """
    # Create a model that maps the input image to the activations of the specified layer
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        layer_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    # Compute gradients with respect to the layer outputs
    gradients = tape.gradient(loss, layer_outputs)

    # Pool the gradients across the channels
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    # Weight the channels by the gradients
    layer_outputs = layer_outputs[0]
    weighted_output = layer_outputs * pooled_gradients

    # Average over all channels
    cam = tf.reduce_sum(weighted_output, axis=-1).numpy()

    # Normalize the heatmap
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    return cam


def create_heatmap_overlay(img, cam, alpha=0.5):
    """
    Create a heatmap overlay on the input image.

    Args:
        img: Input image
        cam: Class activation map
        alpha: Transparency factor

    Returns:
        Overlay image
    """
    # Resize CAM to match input image size
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # Apply jet colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Convert RGB -> BGR for cv2
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Superimpose the heatmap on original image
    superimposed_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img


def visualize_gradcam(model, X_samples, y_samples, layer_name=None):
    """
    Visualize Grad-CAM for sample images.

    Args:
        model: Trained model
        X_samples: Sample images
        y_samples: Sample labels
        layer_name: Name of the layer to use for Grad-CAM
    """
    # If layer_name not provided, find the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.TimeDistributed):
                layer_name = layer.name
                break

    print(f"Using layer: {layer_name} for Grad-CAM")

    # Get predictions
    y_pred_proba = model.predict(X_samples)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Visualize for each sample
    num_samples = len(X_samples)
    plt.figure(figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Get sample
        sample = X_samples[i] if len(X_samples.shape) == 4 else X_samples[i][0]
        true_label = 'Fake' if y_samples[i] == 1 else 'Real'
        pred_label = 'Fake' if y_pred[i] == 1 else 'Real'

        # Compute Grad-CAM
        sample_expanded = np.expand_dims(sample, axis=0)
        cam = compute_gradcam(model, sample_expanded, layer_name, class_idx=y_pred[i])

        # Create heatmap overlay
        heatmap_overlay = create_heatmap_overlay(sample, cam)

        # Display original image
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(sample)
        plt.title(f"Original - True: {true_label}, Pred: {pred_label}")
        plt.axis('off')

        # Display heatmap overlay
        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(heatmap_overlay)
        plt.title(f"Grad-CAM - Confidence: {y_pred_proba[i][0]:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_visualization.png')
    plt.show()