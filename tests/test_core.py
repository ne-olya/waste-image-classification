import numpy as np
import torch
from PIL import Image

from ecosort.gradcam import GradCAM, overlay_heatmap
from ecosort.models import SimpleCNN, target_layer
from ecosort.metrics import ClassificationMeter, expected_calibration_error


def test_model_shape():
    assert SimpleCNN(7)(torch.rand(2, 3, 32, 32)).shape == (2, 7)


def test_gradcam_contract():
    model = SimpleCNN(7)
    cam = GradCAM(model, target_layer(model))
    heatmap, logits = cam(torch.rand(1, 3, 32, 32))
    cam.close()
    assert heatmap.shape == (32, 32) and logits.shape == (1, 7)
    assert 0 <= heatmap.min() <= heatmap.max() <= 1


def test_overlay_shape():
    overlay = overlay_heatmap(Image.new("RGB", (40, 30)), np.ones((10, 10), dtype=np.float32))
    assert overlay.shape == (30, 40, 3)


def test_classification_metrics_and_calibration():
    target = torch.tensor([0, 1])
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    meter = ClassificationMeter(2)
    meter.update(logits, target)
    assert meter.compute()["macro_f1"] == 1
    assert 0 <= expected_calibration_error(logits, target) <= 1
