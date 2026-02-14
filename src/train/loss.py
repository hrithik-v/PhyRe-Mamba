import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, v_pred, x0, x1):
        """
        Rectified Flow Matching Loss.
        Target velocity v_target = x1 - x0 (straight line path).
        Args:
            v_pred: Predicted velocity field (B, 3, H, W).
            x0: Source image (Degraded) (B, 3, H, W).
            x1: Target image (Clean) (B, 3, H, W).
        """
        v_target = x1 - x0
        loss = self.mse(v_pred, v_target)
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super().__init__()
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net='vgg')
            if use_gpu:
                self.loss_fn.cuda()
        except ImportError:
            print("LPIPS library not found. Perceptual loss will be 0.")
            self.loss_fn = None

    def forward(self, pred, target):
        if self.loss_fn is None:
            return torch.tensor(0.0, device=pred.device)
        return self.loss_fn(pred, target).mean()
