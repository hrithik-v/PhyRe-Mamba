# PhyRe-Mamba: Physics-Guided Rectified Mamba Flow

This project implements the PhyRe-Mamba architecture for underwater image restoration using Flow Matching and Mamba State Space Models.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: `mamba-ssm` and `causal-conv1d` require CUDA-compatible versions. Refer to [Mamba's installation guide](https://github.com/state-spaces/mamba) if issues arise.

### 2. Dataset

The project expects the UIEB dataset in the following structure:

```
Dataset/
├── train/
│   ├── input/      # Degraded images
│   └── target/     # Reference/clean images
├── testset(ref)/
│   ├── test-UIEB/
│   │   ├── input/
│   │   └── target/
│   ├── test-EUVP/
│   └── test-LSUI/
└── testset(non-ref)/
    ├── test-UIEB-unpaired/
    ├── test-EUVP-unpaired/
    └── test-RUIE-unpaired/
```

**Default path**: `/kaggle/input/datasets/xuhangc/underwaterbenchmarkdataset/Dataset/`

Update `root_dir` in `config.yaml` if using a different path.

## Usage

### Configuration

All training parameters are in `config.yaml`:

```yaml
# Dataset
root_dir: /kaggle/input/datasets/xuhangc/underwaterbenchmarkdataset/Dataset/train
filename_prefix: uieb
image_size: [256, 256]

# Training
batch_size: 4
lr: 0.0002
epochs: 200
lambda_per: 0.0
val_freq: 1

# Runtime
num_workers: 2
save_dir: ./checkpoints
```

### Architecture Verification

Verify model initialization and forward pass:

```bash
python verify_arch.py
```

### Overfitting Test (Recommended Before Full Training)

Test if the model can overfit on a tiny subset (5 images, 100 epochs) to ensure the training pipeline works:

```bash
python overfit_test.py --config config.yaml --num-images 5 --epochs 100
```

Expected output:
- Loss should decrease from ~0.75 to ~0.003
- PSNR should reach ~25+ dB
- If this succeeds, full training will work

### Full Training

Start training with all images:

```bash
python train.py --config config.yaml
```

The trainer will:
1. Print total UIEB images found, train/val split
2. Log loss and PSNR per epoch during training
3. Log MSE and PSNR per validation epoch (at `val_freq` frequency)
4. Save best model to `./checkpoints/best_model.pth`
5. Save periodic checkpoints every 10 epochs

**Multi-GPU Support**: Automatically detects and uses all available GPUs via `DataParallel` (e.g., 2 GPUs will be used if available).

## Project Structure

```
src/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # UIEBDataset with train/val split
│   ├── physics.py          # Dark channel prior, transmission, depth
│   └── __init__.py
├── model/
│   ├── __init__.py
│   ├── phyre_mamba.py      # Main model
│   ├── encoder.py          # CNN encoder
│   ├── decoder.py          # Cross-latent decoder
│   └── blocks.py           # VSSBlock (Mamba wrapper)
├── train/
│   ├── __init__.py
│   ├── trainer.py          # Training loop with PSNR/MSE logging
│   └── loss.py             # Flow Matching & Perceptual loss
└── utils/
    ├── __init__.py
    └── config.py           # YAML config loader

train.py                     # Main training script (uses config.yaml)
overfit_test.py            # Overfitting test for sanity check
verify_arch.py             # Architecture verification
config.yaml                # Training configuration
requirements.txt           # Dependencies
```

## Key Features

- **Physics-Guided**: Computes transmission and depth maps as additional input channels
- **Flow Matching**: Uses rectified flow for smooth image restoration
- **Mamba Blocks**: Global context modeling via State Space Models
- **Multi-GPU**: Automatic DataParallel support for multiple GPUs
- **PSNR Logging**: Tracks PSNR during training and validation
- **Config-Driven**: All hyperparameters in `config.yaml`
- **Sanity Check**: Overfitting test to verify training pipeline

## Metrics

The trainer logs:
- **Training**: Loss and PSNR per batch (real-time in progress bar)
- **Training Summary**: Average loss and PSNR per epoch
- **Validation**: MSE and PSNR per validation epoch
- **Checkpointing**: Best model saved when val MSE improves

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` in `config.yaml` |
| Slow training | Increase `num_workers` or use multiple GPUs |
| Mamba compilation error | Reinstall `mamba-ssm` with correct CUDA version |
| Dataset not found | Verify `root_dir` in `config.yaml` and folder structure |

## Citation

If you use this code, please cite the PhyRe-Mamba paper (if available) and acknowledge the UIEB dataset.
