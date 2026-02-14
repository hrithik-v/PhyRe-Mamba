import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from .loss import FlowMatchingLoss, PerceptualLoss

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        
        self.criterion_flow = FlowMatchingLoss()
        # Initialize Perceptual Loss lazily or if configured
        self.lambda_per = config.get('lambda_per', 0.0)
        self.criterion_per = None
        if self.lambda_per > 0:
             self.criterion_per = PerceptualLoss(use_gpu=(self.device.type == 'cuda'))
             
        self.save_dir = config.get('save_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        
        for batch in pbar:
            # Inputs
            net_input = batch['net_input'].to(self.device) # (B, 5, H, W)
            x0 = batch['x0'].to(self.device)               # (B, 3, H, W) -- Source
            x1 = batch['x1'].to(self.device)               # (B, 3, H, W) -- Target
            
            # Forward
            v_pred = self.model(net_input)
            
            # Loss
            loss_flow = self.criterion_flow(v_pred, x0, x1)
            loss = loss_flow
            
            # Optional: Perceptual Loss on "One-Step Prediction"
            # Valid only if we assume x1_pred = x0 + v_pred (for t=1)
            # This is a heuristic to guide the flow to be perceptually good at t=1
            if self.criterion_per and self.lambda_per > 0:
                x1_pred = x0 + v_pred
                loss_per = self.criterion_per(x1_pred, x1)
                loss += self.lambda_per * loss_per
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_mse = 0
        with torch.no_grad():
            for batch in self.val_loader:
                net_input = batch['net_input'].to(self.device)
                x0 = batch['x0'].to(self.device)
                x1 = batch['x1'].to(self.device)
                
                # Inference (Euler Step)
                # dt = 1.0 (One step flow)
                v_pred = self.model(net_input)
                x1_pred = x0 + v_pred
                
                mse = torch.nn.functional.mse_loss(x1_pred, x1)
                total_mse += mse.item()
                
        avg_mse = total_mse / len(self.val_loader)
        print(f"Validation Epoch {epoch}: MSE = {avg_mse:.6f}")
        return avg_mse

    def train(self):
        best_mse = float('inf')
        
        for epoch in range(1, self.config['epochs'] + 1):
            train_loss = self.train_epoch(epoch)
            val_mse = self.validate(epoch)
            
            self.scheduler.step()
            
            # Checkpoint
            if val_mse < best_mse:
                best_mse = val_mse
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                
            # Periodic Save
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'epoch_{epoch}.pth'))
