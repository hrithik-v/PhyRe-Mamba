# PhyRe-Mamba: Physics-Guided Rectified Mamba Flow

This project implements the PhyRe-Mamba architecture for underwater image restoration.

## Setup

1.  **Install Dependencies**:
    The project requires PyTorch and Mamba-SSM.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `mamba-ssm` and `causal-conv1d` may require specific CUDA versions. Refer to [Mamba installation guide](https://github.com/state-spaces/mamba) if issues arise.*

2.  **Dataset**:
    Download the [UIEB dataset](https://li-chongyi.github.io/proj_benchmark.html).
    Ensure the structure is:
    ```
    UIEB/
    ├── raw-890/
    └── reference-890/
    ```

## Usage

### Training
To start training the model:
```bash
python train.py --data_root /path/to/UIEB --epochs 100 --batch_size 8 --save_dir ./checkpoints
```

### Architecture Verification
To verify the model architecture without data:
```bash
python verify_arch.py
```

## Project Structure
-   `src/model`: Core PhyRe-Mamba architecture (VSSBlock, CrossLatentDecoder).
-   `src/data`: Data loading and physics map computation (DCP, Depth).
-   `src/train`: Flow Matching loss and training loop.
