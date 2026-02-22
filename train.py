import argparse

from src.data import UIEBDataset
from src.model import PhyReMamba
from src.train import Trainer
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train PhyRe-Mamba")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    root_dir = cfg.get("root_dir")
    if not root_dir:
        raise ValueError("root_dir is required in config.yaml")

    image_size = cfg.get("image_size", [256, 256])
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        image_size = (int(image_size[0]), int(image_size[1]))
    else:
        raise ValueError("image_size must be a list/tuple of [H, W]")

    filename_prefix = cfg.get("filename_prefix", "uieb")

    # Count total UIEB images
    import os
    input_dir = os.path.join(root_dir, 'input')
    all_images = sorted(os.listdir(input_dir)) if os.path.isdir(input_dir) else []
    if filename_prefix:
        prefix = filename_prefix.lower()
        all_images = [n for n in all_images if n.lower().startswith(prefix)]
    total_images = len(all_images)
    print(f"Total {filename_prefix.upper()} images found: {total_images}\n")

    train_dataset = UIEBDataset(
        root_dir=root_dir,
        mode="train",
        size=image_size,
        filename_prefix=filename_prefix
    )
    val_dataset = UIEBDataset(
        root_dir=root_dir,
        mode="val",
        size=image_size,
        filename_prefix=filename_prefix
    )

    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Train images: {len(train_dataset)}")
    print(f"  Val images:   {len(val_dataset)}")
    print(f"  Image size:   {image_size}")
    print(f"{'='*60}\n")

    model = PhyReMamba()

    train_cfg = {
        "batch_size": cfg.get("batch_size", 4),
        "lr": cfg.get("lr", 2e-4),
        "epochs": cfg.get("epochs", 200),
        "lambda_per": cfg.get("lambda_per", 0.0),
        "val_freq": cfg.get("val_freq", 1),
        "save_dir": cfg.get("save_dir", "./checkpoints"),
        "num_workers": cfg.get("num_workers", 2)
    }

    trainer = Trainer(model, train_dataset, val_dataset, train_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
