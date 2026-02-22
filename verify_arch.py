import torch
from src.model import PhyReMamba
from src.train import FlowMatchingLoss

def verify():
    print("Verifying PhyRe-Mamba Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Model Initialization
    try:
        model = PhyReMamba().to(device)
        print("[PASS] Model initialized.")
    except Exception as e:
        print(f"[FAIL] Model initialization failed: {e}")
        return

    # 2. Forward Pass Dimensions
    # Input: (B, 5, 256, 256)
    B, C, H, W = 2, 5, 256, 256
    dummy_input = torch.randn(B, C, H, W, device=device)
    
    try:
        output = model(dummy_input)
        print(f"[PASS] Forward pass successful. Output shape: {output.shape}")
        
        expected_shape = (B, 3, H, W)
        if output.shape == expected_shape:
            print("[PASS] Output shape is correct.")
        else:
            print(f"[FAIL] Output shape mismatch. Expected {expected_shape}, got {output.shape}")
            return
            
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return

    # 3. Loss and Backward Pass
    criterion = FlowMatchingLoss()
    x0 = torch.randn(B, 3, H, W, device=device) # Source
    x1 = torch.randn(B, 3, H, W, device=device) # Target
    
    try:
        loss = criterion(output, x0, x1)
        loss.backward()
        print(f"[PASS] Backward pass successful. Loss: {loss.item()}")
    except Exception as e:
        print(f"[FAIL] Backward pass failed: {e}")
        return

    print("Verification Completed Successfully.")

if __name__ == "__main__":
    verify()
