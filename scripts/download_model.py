#!/usr/bin/env python3
"""Download a fine-tuned model from Together AI.

Usage:
    python scripts/download_model.py ft-ad3a31ac-b7eb
    python scripts/download_model.py ft-ad3a31ac-b7eb --output data/local_models/sft
    python scripts/download_model.py ft-ad3a31ac-b7eb --checkpoint adapter
"""

import argparse
import os
import sys
import tarfile
import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def download_model(ft_id: str, output_dir: Path, checkpoint: str = "merged"):
    try:
        from together import Together
    except ImportError:
        raise ImportError("together package required: pip install together")

    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / f"{ft_id}_{checkpoint}.tar.gz"

    print(f"Downloading '{checkpoint}' checkpoint for job {ft_id}...")
    print("(This may take several minutes for large models)\n")

    resp = client.fine_tuning.content(ft_id=ft_id, checkpoint=checkpoint)
    resp.write_to_file(str(archive_path))

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Downloaded {size_mb:.0f} MB → {archive_path}")

    # Extract
    print("Extracting...")
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tar:
            tar.extractall(output_dir)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(output_dir)
    else:
        print("Unknown archive format — leaving compressed file in place.")
        print(f"Archive saved to: {archive_path}")
        return

    archive_path.unlink()

    # Find extracted model directory (HuggingFace format has config.json)
    model_dir = None
    for candidate in sorted(output_dir.rglob("config.json")):
        model_dir = candidate.parent
        break

    if model_dir:
        print(f"\nModel extracted to: {model_dir}")
        print(f"\nTo use in config:")
        print(f"  rollout_model:")
        print(f"    provider: mlx")
        print(f"    name: {model_dir}")
    else:
        print(f"\nExtracted to: {output_dir}")
        print("(No config.json found — check extracted files manually)")


def main():
    parser = argparse.ArgumentParser(description="Download a fine-tuned model from Together AI")
    parser.add_argument("ft_id", help="Fine-tuning job ID (e.g. ft-ad3a31ac-b7eb)")
    parser.add_argument(
        "--output", default="data/local_models",
        help="Output directory (default: data/local_models)",
    )
    parser.add_argument(
        "--checkpoint", choices=["merged", "adapter"], default="merged",
        help="'merged' = full model (ready to use), 'adapter' = LoRA weights only (default: merged)",
    )
    args = parser.parse_args()

    download_model(args.ft_id, Path(args.output), args.checkpoint)


if __name__ == "__main__":
    main()
