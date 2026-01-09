import argparse
import hashlib
import json
from pathlib import Path

import torch

from .inference import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="exports/ecosort.ts")
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint)
    traced = torch.jit.trace(
        model, torch.zeros(1, 3, checkpoint["image_size"], checkpoint["image_size"])
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "sha256": digest,
        "architecture": checkpoint["architecture"],
        "classes": checkpoint["classes"],
        "image_size": checkpoint["image_size"],
        "temperature": checkpoint.get("temperature", 1),
        "metrics": checkpoint.get("metrics", {}),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
