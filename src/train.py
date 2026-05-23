import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.ingestion import get_dataloaders
from src.models import EmotionCNN

def train_model(epochs=15, batch_size=32, lr=1e-3, weight_decay=1e-4, model_dir="models"):
    print("====================================================")
    print("PHASE 3: DETERMINISTIC MODEL TRAINING START")
    print("====================================================")
    
    # 1. Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # 2. Get DataLoaders
    train_loader, val_loader, _, _, _, _ = get_dataloaders(batch_size=batch_size, val_size=0.2)
    
    # 3. Compile computational graph
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training device bound to: {device}")
    
    model = EmotionCNN(num_classes=7).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Adaptive Scheduler (Cosine Annealing)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        
        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val
        
        # Log progress
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:5.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | LR: {current_lr:.6f}")
        
        # Save Best Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            checkpoint_path = os.path.join(model_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    [OK] Best checkpoint saved to {checkpoint_path} with Val Loss = {val_loss:.4f}, Val Acc = {val_acc*100:5.2f}%")
            
    print("\n[SUCCESS] Deterministic model training complete.")
    print(f"[*] Final Best Validation Accuracy: {best_val_acc*100:.2f}%")
    return history

if __name__ == "__main__":
    train_model(epochs=15)
