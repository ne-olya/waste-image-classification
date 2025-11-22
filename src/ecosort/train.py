import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import class_weights, dataset
from .models import create_model
from .metrics import ClassificationMeter


def epoch(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    total = 0.0
    meter = ClassificationMeter(len(loader.dataset.classes))
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += loss.item() * len(images)
        meter.update(logits.detach(), labels)
    return {"loss": total / len(loader.dataset), **meter.compute()}


def set_backbone_trainable(model, architecture, trainable):
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    head = get_head(model, architecture)
    for parameter in head.parameters():
        parameter.requires_grad = True


def make_optimizer(model, architecture, train_cfg, backbone_enabled):
    head = get_head(model, architecture)
    head_ids = {id(parameter) for parameter in head.parameters()}
    groups = [{"params": list(head.parameters()), "lr": train_cfg["learning_rate"]}]
    backbone = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in head_ids and parameter.requires_grad
    ]
    if backbone_enabled and backbone:
        groups.append({"params": backbone, "lr": train_cfg["backbone_learning_rate"]})
    return torch.optim.AdamW(groups, weight_decay=train_cfg["weight_decay"])


def get_head(model, architecture):
    if architecture.startswith("resnet"):
        return model.fc
    if architecture == "simple_cnn":
        return model.classifier
    return model.classifier[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficientnet.yaml")
    parser.add_argument("--output", default="checkpoints/best.pt")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = dataset(cfg["data"]["root"], "train", cfg["data"]["image_size"], True)
    val_set = dataset(cfg["data"]["root"], "val", cfg["data"]["image_size"])
    loaders = {
        "train": DataLoader(
            train_set,
            cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
        ),
        "val": DataLoader(
            val_set, cfg["data"]["batch_size"], num_workers=cfg["data"]["num_workers"]
        ),
    }
    freeze = cfg["train"]["phase"] in {"head_only", "full_finetune"}
    model = create_model(
        cfg["model"]["architecture"], len(cfg["classes"]), cfg["model"]["pretrained"], freeze
    ).to(device)
    weights = class_weights(train_set).to(device) if cfg["train"]["class_weighted"] else None
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=weights, label_smoothing=cfg["train"]["label_smoothing"]
    )
    optimizer = make_optimizer(model, cfg["model"]["architecture"], cfg["train"], False)
    best, history = -1, []
    for number in range(cfg["train"]["epochs"]):
        if cfg["train"]["phase"] == "full_finetune" and number == cfg["train"]["head_epochs"]:
            set_backbone_trainable(model, cfg["model"]["architecture"], True)
            optimizer = make_optimizer(model, cfg["model"]["architecture"], cfg["train"], True)
        row = {
            "epoch": number + 1,
            "train": epoch(model, loaders["train"], loss_fn, device, optimizer),
            "val": epoch(model, loaders["val"], loss_fn, device),
        }
        history.append(row)
        print(json.dumps(row))
        if row["val"]["macro_f1"] > best:
            best = row["val"]["macro_f1"]
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": cfg["model"]["architecture"],
                    "classes": cfg["classes"],
                    "image_size": cfg["data"]["image_size"],
                    "temperature": 1.0,
                    "metrics": row["val"],
                },
                args.output,
            )


if __name__ == "__main__":
    main()
