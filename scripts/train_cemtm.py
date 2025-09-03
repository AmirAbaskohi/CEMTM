import os
import yaml
import argparse
import torch

from trainer.trainer import train_cemtm


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train CEMTM")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    # Set visible GPU if specified
    device = config["training"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available. Falling back to CPU.")
        config["training"]["device"] = "cpu"

    os.makedirs(config["output"]["save_dir"], exist_ok=True)
    os.makedirs(config["output"]["log_dir"], exist_ok=True)

    print(f"Starting training with config: {args.config}")
    train_cemtm(config)


if __name__ == "__main__":
    main()
