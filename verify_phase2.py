import torch
from src.models import EmotionCNN
from src.tracker import map_emotion_to_action_units
from src.ingestion import EMOTIONS

def run_validation():
    print("====================================================")
    print("PHASE 2 VALIDATION: CORE PIPELINE & GRAPH COMPILATION")
    print("====================================================")
    
    # 1. Compile Computational Graph (Instantiate Model)
    batch_size = 8
    model = EmotionCNN(num_classes=7)
    model.eval()  # Put in evaluation mode
    print("[*] EmotionCNN computational graph successfully compiled.")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total trainable parameters: {total_params:,}")
    
    # 2. Run dummy forward pass
    dummy_input = torch.randn(batch_size, 1, 48, 48)  # B x C x H x W
    print(f"[*] Created dummy input tensor of shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"[*] Dummy output tensor shape: {output.shape}")
    
    # 3. Shape validation (B x 7)
    assert output.shape == (batch_size, 7), f"Error: Output shape must be ({batch_size}, 7), got {output.shape}"
    print("[OK] Forward pass shape matched exactly B x 7.")
    
    # 4. Check probability distribution mapping
    probabilities = torch.softmax(output, dim=1)
    print(f"[*] Sample model softmax output:\n{probabilities[0]}")
    
    sum_probs = probabilities.sum(dim=1)
    print(f"[*] Sum of probabilities across classes: {sum_probs}")
    
    # Assert probabilities sum to approximately 1.0 (with floating point tolerance)
    for p_sum in sum_probs:
        assert torch.allclose(p_sum, torch.tensor(1.0)), f"Error: Softmax probabilities sum to {p_sum.item()}, not 1.0!"
    print("[OK] Softmax outputs form a mathematically valid probability distribution.")
    
    # 5. Verify Action Unit mappings
    print("[*] Verifying Action Unit binding schema:")
    for emotion in EMOTIONS:
        au_mapping = map_emotion_to_action_units(emotion)
        print(f"    - {emotion:10} -> {au_mapping['description']} | Units: {au_mapping['units']}")
        assert len(au_mapping['units']) > 0, f"Error: No Action Units found for emotion {emotion}!"
    print("[OK] Action Unit mappings are correctly bound.")
    
    print("\n====================================================")
    print("PHASE 2 STATE VALIDATION: SUCCESSFUL")
    print("====================================================")

if __name__ == "__main__":
    run_validation()
