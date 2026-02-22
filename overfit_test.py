import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm

from src.data import UIEBDataset
from src.model import PhyReMamba
from src.train.loss import FlowMatchingLoss
from src.utils.config import load_config


def overfit_test(config_path="config.yaml", num_images=5, epochs=100):
    """
    Overfitting test: Train on a tiny subset to verify the model can learn.
    
    Args:
        config_path: Path to YAML config
        num_images: Number of images to use (3-5 recommended)
        epochs: Number of epochs to train (100+ recommended)
    """
    print(f"\n{'='*60}")
    print(f"OVERFITTING TEST")
    print(f"{'='*60}\n")
    
    cfg = load_config(config_path)
    
    # Setup dataset
    root_dir = cfg.get("root_dir")
    if not root_dir:
        raise ValueError("root_dir is required in config.yaml")
    
    image_size = cfg.get("image_size", [256, 256])
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        image_size = (int(image_size[0]), int(image_size[1]))
    
    filename_prefix = cfg.get("filename_prefix", "uieb")
    
    # Load full dataset
    full_dataset = UIEBDataset(
        root_dir=root_dir,
        mode="train",
        size=image_size,
        filename_prefix=filename_prefix
    )
    
    # Create tiny subset
    num_images = min(num_images, len(full_dataset))
    tiny_dataset = Subset(full_dataset, list(range(num_images)))
    
    print(f"Using {num_images} images for overfitting test\n")
    
    dataloader = DataLoader(tiny_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Model
    model = PhyReMamba().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Optimizer (high lr for aggressive learning)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = FlowMatchingLoss()
    
    # Training loop
    print(f"Training for {epochs} epochs...\n")
    print(f"{'Epoch':<8} {'Loss':<12} {'PSNR (dB)':<12}")
    print("-" * 35)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_psnr = 0
        
        for batch in dataloader:
            net_input = batch['net_input'].to(device)
            x0 = batch['x0'].to(device)
            x1 = batch['x1'].to(device)
            
            # Forward
            v_pred = model(net_input)
            x1_pred = x0 + v_pred
            
            # Loss
            loss = criterion(v_pred, x0, x1)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            mse = torch.nn.functional.mse_loss(x1_pred, x1)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            total_psnr += psnr.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_psnr = total_psnr / len(dataloader)
        
        print(f"{epoch:<8} {avg_loss:<12.6f} {avg_psnr:<12.2f}")
        
        # Early indicator: if loss doesn't decrease after 20 epochs, likely a problem
        if epoch == 20 and avg_loss > 0.1:
            print("\n⚠️  WARNING: Loss not decreasing significantly after 20 epochs!")
            print("   Check model architecture, loss function, or data loading.")
    
    print("-" * 35)
    print(f"\n✓ Overfitting test completed!")
    print(f"  Final Loss: {avg_loss:.6f}")
    print(f"  Final PSNR: {avg_psnr:.2f} dB")
    
    if avg_psnr > 30:
        print(f"  Status: ✓ Model is learning well! Ready for full training.")
    elif avg_psnr > 20:
        print(f"  Status: ⚠️  Model is learning but slowly. Check hyperparameters.")
    else:
        print(f"  Status: ✗ Model not learning. Check code for bugs.")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overfitting test for PhyRe-Mamba")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to overfit on")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()
    
    overfit_test(config_path=args.config, num_images=args.num_images, epochs=args.epochs)
