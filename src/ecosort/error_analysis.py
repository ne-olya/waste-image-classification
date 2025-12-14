import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .data import dataset
from .inference import load_checkpoint, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="reports/errors")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    test_data = dataset(args.data, "test", checkpoint["image_size"])
    errors = []
    with torch.inference_mode():
        for index, (tensor, target) in enumerate(test_data):
            probabilities = (
                model(tensor.unsqueeze(0).to(device)) / checkpoint.get("temperature", 1.0)
            ).softmax(1)[0]
            predicted = int(probabilities.argmax())
            if predicted != target:
                errors.append(
                    {
                        "index": index,
                        "path": test_data.samples[index][0],
                        "true": checkpoint["classes"][target],
                        "predicted": checkpoint["classes"][predicted],
                        "confidence": float(probabilities[predicted]),
                    }
                )
    errors.sort(key=lambda row: row["confidence"], reverse=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(errors[: args.limit], 1):
        image = Image.open(row["path"]).convert("RGB")
        explanation = predict(model, image, checkpoint["classes"], checkpoint["image_size"], device)
        filename = f"{rank:03d}_{row['true']}_as_{row['predicted']}.jpg"
        Image.fromarray(explanation.heatmap).save(output / filename)
        row["gradcam"] = filename
    (output / "errors.json").write_text(
        json.dumps(errors[: args.limit], ensure_ascii=False, indent=2)
    )
    print(f"Saved {min(len(errors), args.limit)} of {len(errors)} errors to {output}")


if __name__ == "__main__":
    main()
