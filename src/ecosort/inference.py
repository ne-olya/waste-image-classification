from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from . import CLASSES, DISPOSAL
from .data import transforms_for
from .gradcam import GradCAM, overlay_heatmap
from .models import target_layer


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    heatmap: np.ndarray
    recommendation: str


def load_checkpoint(path, device="cpu"):
    from .models import create_model

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = create_model(checkpoint["architecture"], len(checkpoint["classes"]), False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.temperature = float(checkpoint.get("temperature", 1.0))
    model.eval()
    return model, checkpoint


def predict(model, image: Image.Image, classes=CLASSES, size=224, device="cpu"):
    tensor = transforms_for(size)(image.convert("RGB")).unsqueeze(0).to(device)
    cam = GradCAM(model, target_layer(model))
    heatmap, logits = cam(tensor)
    cam.close()
    temperature = getattr(model, "temperature", 1.0)
    probs = (logits / temperature).softmax(1)[0].cpu().numpy()
    index = int(probs.argmax())
    label = classes[index]
    return Prediction(
        label,
        float(probs[index]),
        dict(zip(classes, probs.astype(float))),
        overlay_heatmap(image, heatmap),
        DISPOSAL.get(label, DISPOSAL["other"]),
    )
