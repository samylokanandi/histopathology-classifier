"""
gradcam.py
A from-scratch Grad-CAM implementation (Selvaraju et al., 2017).

The idea: to see which pixels mattered for a CNN's prediction, look at the
gradient of the predicted class score with respect to the feature maps of a
chosen convolutional layer. Average those gradients per channel to get
"importance weights," then take a weighted sum of the feature maps. Channels
that mattered more for this prediction contribute more to the heatmap.

This works on any CNN — you just need to point it at a target conv layer
(for ResNet18, that's typically `model.layer4[-1]`, the last block before
the global average pool).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: shape (1, C, H, W), already preprocessed/normalized.
        target_class: int index to explain. If None, uses the model's top prediction.

        Returns: (heatmap as a (H, W) numpy array scaled 0-1, predicted_class_idx, confidence)
        """
        self.model.eval()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        self.model.zero_grad()
        # Backprop only the score for the class we care about.
        score = output[0, target_class]
        score.backward()

        # gradients/activations shape: (1, num_channels, h, w)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # importance per channel
        weighted_activations = (weights * self.activations).sum(dim=1).squeeze()  # (h, w)

        heatmap = F.relu(weighted_activations).cpu().numpy()  # only positive influence
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap, target_class, confidence


def overlay_heatmap(heatmap, original_image_rgb, alpha=0.45):
    """
    heatmap: (h, w) array in [0, 1] from GradCAM.generate()
    original_image_rgb: (H, W, 3) uint8 array, the input image at its display resolution
    Returns: (H, W, 3) uint8 array — original image with a jet-colormap heatmap overlay
    """
    h, w = original_image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (heatmap_color * alpha + original_image_rgb * (1 - alpha)).astype(np.uint8)
    return overlay
