from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def create_model(name, num_classes=7, pretrained=True, freeze_backbone=False):
    if name == "simple_cnn":
        model = SimpleCNN(num_classes)
        if freeze_backbone:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
        return model
    from torchvision import models

    weights = "DEFAULT" if pretrained else None
    if name.startswith("resnet"):
        model = getattr(models, name)(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head = model.fc
    elif name.startswith("efficientnet") or name.startswith("mobilenet"):
        model = getattr(models, name)(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head = model.classifier[-1]
    else:
        raise ValueError(f"Unknown architecture: {name}")
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
    return model


def target_layer(model):
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("No Grad-CAM target layer for model")
