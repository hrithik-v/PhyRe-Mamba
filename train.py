import argparse
import torch
from src.data import UIEBDataset
from src.model import PhyReMamba
from src.train import Trainer
from src.utils import config # Placeholder, we might just use argparse for now

def main():
    parser = argparse.ArgumentParser(description="Train PhyRe-Mamba")
    parser.add_argument('--data_root', type=str, required=True, help='Path to UIEB dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--lambda_per', type=float, default=0.1, help='Perceptual loss weight')
    
    args = parser.parse_args()
    
    # Config dict
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'save_dir': args.save_dir,
        'lambda_per': args.lambda_per
    }
    
    # Dataset
    # Assuming standard UIEB split for now or just using same dir for train/val split in Dataset class
    train_dataset = UIEBDataset(args.data_root, mode='train')
    val_dataset = UIEBDataset(args.data_root, mode='val')
    
    # Model
    model = PhyReMamba()
    
    # Trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)
    
    # Start Training
    trainer.train()

if __name__ == '__main__':
    main()
