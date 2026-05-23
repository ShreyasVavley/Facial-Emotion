import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.ingestion import get_dataloaders, EMOTIONS
from src.models import EmotionCNN

def evaluate_model(model_path="models/best_model.pth"):
    print("====================================================")
    print("PHASE 3: DETAILED MODEL EVALUATION & CROSS-REVIEW LOG")
    print("====================================================")
    
    # 1. Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}! Please train the model first.")
        return None
        
    # 2. Get DataLoaders
    _, val_loader, _, _, _, _ = get_dataloaders(batch_size=32, val_size=0.2)
    
    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 4. Predict
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 5. Compute Metrics
    report = classification_report(
        all_labels, all_preds, target_names=EMOTIONS, output_dict=True, zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # 6. Structured Output
    print("\n----------------------------------------------------")
    print("AI METRICS LOG SUMMARY:")
    print("----------------------------------------------------")
    print(f"Overall Accuracy: {report['accuracy']*100:.2f}%")
    print(f"Macro Precision:  {report['macro avg']['precision']:.4f}")
    print(f"Macro Recall:     {report['macro avg']['recall']:.4f}")
    print(f"Macro F1-Score:   {report['macro avg']['f1-score']:.4f}")
    print("----------------------------------------------------")
    
    print("\nCLASS-LEVEL PERFORMANCES:")
    print(f"{'Class (Emotion)':15} | {'Precision':10} | {'Recall':10} | {'F1-Score':10}")
    print("-" * 55)
    for emotion in EMOTIONS:
        metrics = report[emotion]
        print(f"{emotion:15} | {metrics['precision']:.4f}     | {metrics['recall']:.4f}   | {metrics['f1-score']:.4f}")
        
    print("\nCONFUSION MATRIX:")
    print(conf_matrix)
    
    return report

if __name__ == "__main__":
    evaluate_model()
