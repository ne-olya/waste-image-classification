from pathlib import Path

import torch
from torchvision import datasets, transforms


def transforms_for(size=224, train=False):
    ops = [transforms.Resize((size, size))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomPerspective(0.15, p=0.25),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


def dataset(root, split, size=224, train=False):
    return datasets.ImageFolder(Path(root) / split, transform=transforms_for(size, train))


def class_weights(image_folder):
    counts = torch.bincount(
        torch.tensor(image_folder.targets), minlength=len(image_folder.classes)
    ).float()
    return counts.sum() / (len(counts) * counts.clamp_min(1))
