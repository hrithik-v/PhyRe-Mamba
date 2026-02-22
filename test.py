import argparse
import os
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set backend to Agg for headless environments like Kaggle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.model import PhyReMamba
from src.data.dataset import UIEBDataset
from src.utils.config import load_config

def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor (C, H, W) in [0, 1] range back to 
    an RGB numpy array (H, W, 3) in [0, 255] for saving/plotting.
    """
    # Detach and move to CPU, then numpy
    img_np = tensor.detach().cpu().numpy()
    
    # Transpose from (C, H, W) back to (H, W, C)
    img_np = img_np.transpose(1, 2, 0)
    
    # Clip to [0, 1] just in case of slight over/under shoots from the model
    img_np = np.clip(img_np, 0.0, 1.0)
    
    # Convert to uint8 [0, 255]
    img_np = (img_np * 255.0).astype(np.uint8)
    
    return img_np


def run_inference(config_path, checkpoint_path, num_samples, out_dir):
    print(f"\n{'='*60}")
    print(f"PHYRE-MAMBA INFERENCE")
    print(f"{'='*60}\n")
    
    # Load config and determine image size
    cfg = load_config(config_path)
    image_size = cfg.get("image_size", [256, 256])
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        image_size = (int(image_size[0]), int(image_size[1]))
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Model
    print("Initializing model...")
    model = PhyReMamba()
    
    # Load Checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    model = model.to(device)
    model.eval()
    
    # Load Dataset (Using 'val' mode which now points to test-UIEB if available)
    print("Loading test dataset...")
    test_dataset = UIEBDataset(
        root_dir=cfg.get("root_dir"),
        mode="val",
        size=image_size,
        filename_prefix=cfg.get("filename_prefix")
    )
    
    total_available = len(test_dataset)
    print(f"Total test images available: {total_available}")
    
    samples_to_process = min(num_samples, total_available)
    print(f"Processing {samples_to_process} sample images...\n")
    
    # DataLoader snippet (shuffle=True to get random samples if we ask for < Total)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    processed_count = 0
    all_inputs = []
    all_outputs = []
    all_gts = []
    
    with torch.no_grad():
        for batch in test_loader:
            if processed_count >= samples_to_process:
                break
                
            net_input = batch['net_input'].to(device)
            x0 = batch['x0'].to(device)
            
            # Predict Flow (Velocity)
            v_pred = model(net_input)
            
            # Euler Integration step (dt = 1.0)
            x1_pred = x0 + v_pred
            
            # Convert tensors to displayable numpy arrays
            img_in = tensor_to_image(x0[0])
            img_out = tensor_to_image(x1_pred[0])
            
            all_inputs.append(img_in)
            all_outputs.append(img_out)
            
            # We also get the Ground Truth (x1) to display if desired
            if 'x1' in batch:
                x1_gt = batch['x1'].to(device)
                img_gt = tensor_to_image(x1_gt[0])
                all_gts.append(img_gt)
                has_gt = True
            else:
                has_gt = False
            
            print(f"  [{processed_count+1}/{samples_to_process}] Processed image.")
            processed_count += 1
            
    # Create a single combined grid plot
    cols = 3 if has_gt else 2
    rows = samples_to_process
    
    fig_width = 12 if has_gt else 8
    fig_height = 4 * rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Handle the case where we only sample 1 image (axes is 1D instead of 2D)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i in range(rows):
        # Input
        axes[i, 0].imshow(all_inputs[i])
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Input (Degraded)", fontsize=16, pad=10)
            
        # Output
        axes[i, 1].imshow(all_outputs[i])
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title("Output (Restored)", fontsize=16, pad=10)
            
        # Ground Truth
        if has_gt:
            axes[i, 2].imshow(all_gts[i])
            axes[i, 2].axis("off")
            if i == 0:
                axes[i, 2].set_title("Ground Truth", fontsize=16, pad=10)
                
    plt.tight_layout()
    
    # Save the huge combined figure
    out_filename = f"combined_results_{samples_to_process}samples.png"
    save_path = os.path.join(out_dir, out_filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"\n✓ Completed! Grid saved to: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PhyRe-Mamba on test images.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--checkpoint", default="./checkpoints/best_model.pth", help="Path to model weights.")
    parser.add_argument("--samples", type=int, default=10, help="Number of random images to process and save.")
    parser.add_argument("--out", default="./inference_results", help="Directory to save the side-by-side images.")
    
    args = parser.parse_args()
    
    run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_samples=args.samples,
        out_dir=args.out
    )
