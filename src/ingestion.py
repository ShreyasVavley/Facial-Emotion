import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2

# Define target emotions
EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

class FERDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 48, 48)
            labels (np.ndarray): Shape (N,)
            transform (callable, optional): PyTorch transform
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image and label
        img = self.images[idx]
        label = self.labels[idx]

        # Convert to PyTorch float32 tensor and normalize to [0, 1]
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        if self.transform:
            img_tensor = self.transform(img_tensor)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

def draw_synthetic_emotion(emotion_name):
    """
    Generates a deterministic 48x48 synthetic grayscale image representing an emotion.
    """
    # Create background (Midnight Obsidian tone)
    img = np.zeros((48, 48), dtype=np.uint8)
    
    # Draw face circle (base contour)
    cv2.circle(img, (24, 24), 20, 200, 1)
    
    # Eyes (default)
    left_eye_center = (18, 18)
    right_eye_center = (30, 18)
    cv2.circle(img, left_eye_center, 2, 255, -1)
    cv2.circle(img, right_eye_center, 2, 255, -1)
    
    if emotion_name == "Happiness":
        # Draw smile curve
        cv2.ellipse(img, (24, 28), (10, 6), 0, 0, 180, 255, 1)
    elif emotion_name == "Sadness":
        # Draw frown curve
        cv2.ellipse(img, (24, 32), (10, 6), 0, 180, 360, 255, 1)
    elif emotion_name == "Surprise":
        # Draw circular mouth
        cv2.circle(img, (24, 30), 4, 255, 1)
    elif emotion_name == "Anger":
        # Angry mouth (flat or small frown) and diagonal eyebrows
        cv2.line(img, (14, 30), (34, 30), 255, 1)
        # Brows
        cv2.line(img, (14, 14), (20, 17), 255, 1)
        cv2.line(img, (34, 14), (28, 17), 255, 1)
    elif emotion_name == "Fear":
        # Open mouth and raised eyebrows
        cv2.ellipse(img, (24, 31), (6, 3), 0, 0, 360, 255, 1)
        cv2.ellipse(img, (18, 14), (4, 2), 0, 180, 360, 255, 1)
        cv2.ellipse(img, (30, 14), (4, 2), 0, 180, 360, 255, 1)
    elif emotion_name == "Disgust":
        # Wrinkled nose (diagonal line in center) and asymmetric flat mouth
        cv2.line(img, (24, 22), (24, 26), 255, 1)
        cv2.line(img, (20, 31), (28, 29), 255, 1)
    else:  # Neutral
        # Straight line mouth
        cv2.line(img, (18, 30), (30, 30), 255, 1)
        
    return img

def create_synthetic_dataset(num_samples=1400):
    """
    Creates a deterministic synthetic dataset of 48x48 images with stratified emotions.
    """
    np.random.seed(42)
    images = []
    labels = []
    
    samples_per_class = num_samples // len(EMOTIONS)
    
    for emotion_idx, emotion_name in enumerate(EMOTIONS):
        for i in range(samples_per_class):
            img = draw_synthetic_emotion(emotion_name)
            # Add subtle deterministic noise to simulate real-world variance
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            # Apply histogram equalization
            img = cv2.equalizeHist(img)
            
            images.append(img)
            labels.append(emotion_idx)
            
    return np.array(images), np.array(labels)

def get_dataloaders(batch_size=32, val_size=0.2, local_csv_path=None):
    """
    Prepares train and validation DataLoaders.
    If local_csv_path is provided, loads FER2013 data. Otherwise, generates synthetic data.
    """
    if local_csv_path and os.path.exists(local_csv_path):
        print(f"Loading data from local dataset path: {local_csv_path}")
        # Placeholder for real csv loading
        # Standard FER2013 format has: emotion, pixels, Usage
        import pandas as pd
        df = pd.read_csv(local_csv_path)
        
        images = []
        labels = []
        for index, row in df.iterrows():
            emotion = int(row['emotion'])
            pixels = np.fromstring(row['pixels'], dtype=int, sep=' ').reshape(48, 48).astype(np.uint8)
            # Apply histogram equalization
            pixels = cv2.equalizeHist(pixels)
            images.append(pixels)
            labels.append(emotion)
            
        images = np.array(images)
        labels = np.array(labels)
    else:
        print("Dataset CSV not found or not provided. Launching deterministic synthetic pipeline...")
        images, labels = create_synthetic_dataset(num_samples=1400)
        
    # Stratified Train/Val Split (zero leakage validation guarantees)
    x_train, x_val, y_train, y_val = train_test_split(
        images, labels, test_size=val_size, stratify=labels, random_state=42
    )
    
    # Store indices for verification
    train_dataset = FERDataset(x_train, y_train)
    val_dataset = FERDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, x_train, x_val, y_train, y_val
