import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.activations, self.gradients = model, None, None
        self.handles = [
            layer.register_forward_hook(self._activation),
            layer.register_full_backward_hook(self._gradient),
        ]

    def _activation(self, _module, _inputs, output):
        self.activations = output

    def _gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, tensor, class_index=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        class_index = (
            logits.argmax(1)
            if class_index is None
            else torch.as_tensor([class_index], device=logits.device)
        )
        logits.gather(1, class_index[:, None]).sum().backward()
        weights = self.gradients.mean((2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach()

    def close(self):
        for handle in self.handles:
            handle.remove()


def overlay_heatmap(image: Image.Image, heatmap, alpha=0.45):
    heatmap = np.asarray(Image.fromarray((heatmap * 255).astype("uint8")).resize(image.size)) / 255
    color = np.stack([heatmap, np.zeros_like(heatmap), 1 - heatmap], axis=-1) * 255
    return ((1 - alpha) * np.asarray(image.convert("RGB")) + alpha * color).astype("uint8")
