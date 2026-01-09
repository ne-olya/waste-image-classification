import argparse

import torch
from torch.utils.data import DataLoader

from .data import dataset
from .inference import load_checkpoint
from .metrics import expected_calibration_error


def fit_temperature(logits, labels, steps=100):
    log_temperature = torch.nn.Parameter(torch.zeros((), device=logits.device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=steps)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    loader = DataLoader(dataset(args.data, "val", checkpoint["image_size"]), batch_size=64)
    logits, labels = [], []
    with torch.inference_mode():
        for images, target in loader:
            logits.append(model(images.to(device)))
            labels.append(target.to(device))
    logits, labels = torch.cat(logits), torch.cat(labels)
    before = expected_calibration_error(logits, labels)
    temperature = fit_temperature(logits, labels)
    after = expected_calibration_error(logits / temperature, labels)
    checkpoint["temperature"] = temperature
    checkpoint["calibration"] = {"ece_before": before, "ece_after": after}
    torch.save(checkpoint, args.checkpoint)
    print(checkpoint["calibration"] | {"temperature": temperature})


if __name__ == "__main__":
    main()
