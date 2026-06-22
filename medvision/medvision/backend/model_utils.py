"""
model_utils.py
Loads the trained checkpoint and runs prediction + Grad-CAM for a single uploaded image.
"""

import base64
import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from gradcam import GradCAM, overlay_heatmap

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    gradcam = GradCAM(model, target_layer=model.layer4[-1])
    return model, gradcam, classes


def predict_with_heatmap(image_bytes, model, gradcam, classes, device):
    # Load the image twice at different stages: once normalized for the model,
    # once at display resolution (just resized, not normalized) for the overlay.
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    display_image = pil_image.resize((IMG_SIZE, IMG_SIZE))
    display_array = np.array(display_image)

    input_tensor = _preprocess(pil_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(False)

    heatmap, class_idx, confidence = gradcam.generate(input_tensor)
    overlay = overlay_heatmap(heatmap, display_array)

    overlay_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    heatmap_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "prediction": classes[class_idx],
        "confidence": round(confidence, 4),
        "heatmap_base64": heatmap_base64,
    }
