import sys
import numpy as np
from src.ingestion import get_dataloaders, EMOTIONS

def run_validation():
    print("====================================================")
    print("PHASE 1 VALIDATION: DATA ARCHITECTURE & INGESTION")
    print("====================================================")
    
    # 1. Ingest datasets
    batch_size = 32
    train_loader, val_loader, x_train, x_val, y_train, y_val = get_dataloaders(batch_size=batch_size, val_size=0.2)
    
    # 2. Verify dataset shapes
    print(f"[*] Training images shape: {x_train.shape}")
    print(f"[*] Validation images shape: {x_val.shape}")
    assert x_train.shape[1:] == (48, 48), f"Error: Training image shape must be (48, 48), got {x_train.shape[1:]}"
    assert x_val.shape[1:] == (48, 48), f"Error: Validation image shape must be (48, 48), got {x_val.shape[1:]}"
    print("[OK] Image resolution is verified at 48x48 pixels.")
    
    # 3. Check DataLoader batch shapes (B x 1 x 48 x 48)
    sample_imgs, sample_labels = next(iter(train_loader))
    print(f"[*] DataLoader batch images tensor shape: {sample_imgs.shape}")
    print(f"[*] DataLoader batch labels tensor shape: {sample_labels.shape}")
    
    assert len(sample_imgs.shape) == 4, "Error: Tensor must have 4 dimensions (B x C x H x W)"
    assert sample_imgs.shape[1] == 1, f"Error: Channel dimension must be 1 (Grayscale), got {sample_imgs.shape[1]}"
    assert sample_imgs.shape[2:] == (48, 48), f"Error: Resolution must be 48x48, got {sample_imgs.shape[2:]}"
    assert sample_imgs.shape[0] == batch_size, f"Error: Batch dimension must be {batch_size}, got {sample_imgs.shape[0]}"
    print("[OK] Tensor bounds are validated at B x 1 x 48 x 48.")
    
    # 4. Check normalization range [0.0, 1.0]
    min_val = sample_imgs.min().item()
    max_val = sample_imgs.max().item()
    print(f"[*] Grayscale range: [{min_val:.4f}, {max_val:.4f}]")
    assert 0.0 <= min_val <= max_val <= 1.0, f"Error: Grayscale normalization outside [0, 1] range: [{min_val}, {max_val}]"
    print("[OK] Pixel normalization is successfully bound to [0.0, 1.0].")
    
    # 5. Cryptographic / Set-theoretic partition leakage check (Zero leakage validation)
    # Convert image grids to flat 1D arrays for cryptographic hashing/string match
    flat_train = set(tuple(img.flatten()) for img in x_train)
    flat_val = set(tuple(img.flatten()) for img in x_val)
    
    overlap = flat_train.intersection(flat_val)
    print(f"[*] Number of overlapping samples: {len(overlap)}")
    
    assert len(overlap) == 0, f"CRITICAL LEAKAGE DETECTED: {len(overlap)} images overlap between training and validation sets!"
    print("[OK] Zero leakage across training and validation partitions mathematically verified.")
    
    # 6. Verify class stratification
    train_class_counts = np.bincount(y_train)
    val_class_counts = np.bincount(y_val)
    
    print("[*] Stratified distribution:")
    for idx, name in enumerate(EMOTIONS):
        print(f"    Class {idx} ({name:10}): Train count = {train_class_counts[idx]}, Val count = {val_class_counts[idx]}")
        
    # Check that proportions are extremely similar
    for idx in range(len(EMOTIONS)):
        train_prop = train_class_counts[idx] / len(y_train)
        val_prop = val_class_counts[idx] / len(y_val)
        assert abs(train_prop - val_prop) < 0.05, f"Stratification discrepancy in class {idx}: {train_prop} vs {val_prop}"
    print("[OK] Class distributions are properly stratified.")
    
    print("\n====================================================")
    print("PHASE 1 STATE VALIDATION: SUCCESSFUL")
    print("====================================================")
    
if __name__ == "__main__":
    run_validation()
