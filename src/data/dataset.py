import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from .physics import compute_physics_maps
from PIL import Image

class UIEBDataset(Dataset):
    def __init__(self, root_dir, mode='train', size=(256, 256)):
        """
        Args:
            root_dir: Path to UIEB dataset root. Expected structure:
                      root_dir/raw-890/ (input images)
                      root_dir/reference-890/ (ground truth)
            mode: 'train' or 'val' or 'test'
            size: Tuple (H, W) to resize images to.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.size = size
        
        # Paths setup - Adjust based on actual unzipped structure
        # Assuming standard UIEB structure
        self.input_dir = os.path.join(root_dir, 'raw-890')
        self.gt_dir = os.path.join(root_dir, 'reference-890')
        
        self.image_names = sorted(os.listdir(self.input_dir))
        
        # Simple split for train/val
        # First 800 train, rest val/test (as per report)
        if mode == 'train':
            self.image_names = self.image_names[:800]
        elif mode == 'val':
            self.image_names = self.image_names[800:]
        
        # Filter to ensure matching GT exists (some raw images in UIEB don't have GT)
        self.valid_pairs = []
        for name in self.image_names:
            if os.path.exists(os.path.join(self.gt_dir, name)):
                self.valid_pairs.append(name)
            # handle png vs jpg extension mismatch if any (UIEB is mostly consistent)
            
        print(f"[{mode.upper()}] Loaded {len(self.valid_pairs)} pairs from {root_dir}")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        name = self.valid_pairs[idx]
        input_path = os.path.join(self.input_dir, name)
        gt_path = os.path.join(self.gt_dir, name)
        
        # Load images
        # using cv2 for easy resizing and array manipulation
        img_in = cv2.imread(input_path)
        img_gt = cv2.imread(gt_path)
        
        # Convert BGR to RGB
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        
        # Resize
        img_in = cv2.resize(img_in, self.size)
        img_gt = cv2.resize(img_gt, self.size)
        
        # Normalize to [0, 1]
        img_in = img_in.astype(np.float32) / 255.0
        img_gt = img_gt.astype(np.float32) / 255.0
        
        # Compute Physics Maps (Transmission & Depth)
        # Done on-the-fly. For speed, these should be pre-computed and saved, 
        # but for simplicity/portability we compute here.
        transmission, depth = compute_physics_maps(img_in)
        
        # Expand dims for concatenation: (H, W) -> (H, W, 1)
        transmission = transmission[..., np.newaxis]
        depth = depth[..., np.newaxis]
        
        # Create 5-channel input: R, G, B, T, D
        # Shape: (H, W, 5) -> Transpose to (5, H, W) for PyTorch
        input_tensor = np.concatenate([img_in, transmission, depth], axis=2)
        
        # To Tensor
        input_tensor = torch.from_numpy(input_tensor.transpose(2, 0, 1)).float()
        gt_tensor = torch.from_numpy(img_gt.transpose(2, 0, 1)).float()
        
        # Rectified Flow Targets
        # Source (x0) = input_tensor (first 3 channels usually, but here we condition on it)
        # Actually for flow matching: 
        # Source distribution pi_0 is usually noise or the degraded image itself. 
        # In restoration (I2I), we map Degraded -> Clean.
        # So x0 = img_in, x1 = img_gt.
        # The model conditions on the 5-channel input to predict the velocity field.
        # We return both so the loss can compute v_target = x1 - x0.
        
        # For the model input, we pass the 5-channel tensor.
        # For the flow target, we need the 3-channel RGB components.
        degraded_rgb = input_tensor[:3, :, :]
        
        return {
            'net_input': input_tensor,      # (5, H, W) - Condition
            'x0': degraded_rgb,             # (3, H, W) - Source state
            'x1': gt_tensor,                # (3, H, W) - Target state
            'filename': name
        }
