import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import dataset
from .inference import load_checkpoint
from .metrics import expected_calibration_error


def main():
    from sklearn.metrics import classification_report, confusion_matrix

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="reports/metrics.json")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    loader = DataLoader(dataset(args.data, "test", checkpoint["image_size"]), batch_size=32)
    actual, predicted, all_logits, start = [], [], [], time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            actual += labels.tolist()
            logits = model(images.to(device)) / checkpoint.get("temperature", 1.0)
            all_logits.append(logits.cpu())
            predicted += logits.argmax(1).cpu().tolist()
    elapsed = (time.perf_counter() - start) * 1000 / len(actual)
    result = classification_report(
        actual, predicted, target_names=checkpoint["classes"], output_dict=True, zero_division=0
    )
    result["confusion_matrix"] = confusion_matrix(actual, predicted).tolist()
    result["latency_ms_per_image"] = elapsed
    result["model_size_mb"] = Path(args.checkpoint).stat().st_size / 1024**2
    result["ece"] = expected_calibration_error(torch.cat(all_logits), torch.tensor(actual))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
