import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from .physics import compute_physics_maps
from PIL import Image

class UIEBDataset(Dataset):
    def __init__(self, root_dir='/kaggle/input/datasets/xuhangc/underwaterbenchmarkdataset/Dataset/train', mode='train', size=(256, 256), filename_prefix='uieb'):
        """
        Args:
            root_dir: Path to UIEB dataset root. Expected structure:
                      root_dir/raw-890/ (input images)
                      root_dir/reference-890/ (ground truth)
                      OR
                      root_dir/input/ (input images)
                      root_dir/target/ (ground truth)
            mode: 'train' or 'val' or 'test'
            size: Tuple (H, W) to resize images to.
            filename_prefix: Optional filename prefix filter (case-insensitive).
        """
        self.root_dir = root_dir
        self.mode = mode
        self.size = size

        # Paths setup - Auto-detect common UIEB layouts
        raw_dir = os.path.join(root_dir, 'raw-890')
        ref_dir = os.path.join(root_dir, 'reference-890')
        input_dir = os.path.join(root_dir, 'input')
        target_dir = os.path.join(root_dir, 'target')
        train_input_dir = os.path.join(root_dir, 'train', 'input')
        train_target_dir = os.path.join(root_dir, 'train', 'target')

        if os.path.isdir(raw_dir) and os.path.isdir(ref_dir):
            self.input_dir = raw_dir
            self.gt_dir = ref_dir
        elif os.path.isdir(input_dir) and os.path.isdir(target_dir):
            self.input_dir = input_dir
            self.gt_dir = target_dir
        elif os.path.isdir(train_input_dir) and os.path.isdir(train_target_dir):
            self.input_dir = train_input_dir
            self.gt_dir = train_target_dir
        else:
            raise FileNotFoundError(
                "UIEB dataset not found. Expected raw-890/reference-890 or input/target folders. "
                f"Given root_dir={root_dir}"
            )
        
        self.image_names = sorted(os.listdir(self.input_dir))
        
        # 1. Filter out prefix if provided
        if filename_prefix:
            prefix = filename_prefix.lower()
            self.image_names = [n for n in self.image_names if n.lower().startswith(prefix)]
            
        # 2. Filter out duplicates with "(copy)" in the name
        self.image_names = [n for n in self.image_names if "(copy)" not in n]
        
        # 3. If mode is "val", we need to override the paths to point to the test set
        if mode == 'val':
            # Base directory for the testset is expected to be adjacent to 'train'
            # Assuming root_dir is something like .../Dataset/train
            base_dataset_dir = os.path.dirname(self.root_dir)
            test_dir = os.path.join(base_dataset_dir, 'testset(ref)', 'test-UIEB')
            
            if os.path.isdir(test_dir):
                self.input_dir = os.path.join(test_dir, 'input')
                self.gt_dir = os.path.join(test_dir, 'target')
                if os.path.isdir(self.input_dir):
                    self.image_names = sorted(os.listdir(self.input_dir))
                    if filename_prefix:
                        self.image_names = [n for n in self.image_names if n.lower().startswith(prefix)]
                    self.image_names = [n for n in self.image_names if "(copy)" not in n]
            else:
                print(f"Warning: Test directory not found at {test_dir}. Falling back to splitting train.")
                self.image_names = self.image_names[800:]
        elif mode == 'train':
            # Ensure we only take the first 800 (though if duplicates are filtered, it should be exactly 800)
            self.image_names = self.image_names[:800]
        
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
