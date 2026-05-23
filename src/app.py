import os
import base64
import numpy as np
import cv2
import torch
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
from transformers import pipeline
from src.tracker import detect_faces, preprocess_face, map_emotion_to_action_units
from src.ingestion import EMOTIONS
from src.models import EmotionCNN

app = FastAPI(title="Midnight Obsidian FER Pipeline Dashboard")

# Hugging Face Model configuration
MODEL_NAME = "dima806/facial_emotions_image_detection"
LABEL_MAPPING = {
    "angry": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happiness",
    "sad": "Sadness",
    "surprise": "Surprise",
    "neutral": "Neutral"
}

emotion_classifier = None
local_cnn_model = None
device = None

@app.on_event("startup")
def load_pipeline_model():
    global emotion_classifier, local_cnn_model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = 0 if torch.cuda.is_available() else -1
    
    # Load Hugging Face ViT Model
    print(f"[*] Loading state-of-the-art Hugging Face ViT classifier: {MODEL_NAME} ...")
    try:
        emotion_classifier = pipeline(
            "image-classification",
            model=MODEL_NAME,
            device=device_id
        )
        print("[SUCCESS] State-of-the-art FER2013 Vision Transformer loaded and ready.")
    except Exception as e:
        print(f"[ERROR] Failed to load ViT model: {e}")
        
    # Load Local PyTorch CNN Model
    print("[*] Loading local PyTorch CNN Model architecture...")
    try:
        local_cnn_model = EmotionCNN(num_classes=7).to(device)
        model_path = "models/best_model.pth"
        if os.path.exists(model_path):
            local_cnn_model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[SUCCESS] Local CNN weights loaded from {model_path}.")
        else:
            print(f"[WARNING] Local CNN weights NOT found at {model_path}. Running with uninitialized weights.")
        local_cnn_model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load local PyTorch CNN model: {e}")

class FrameData(BaseModel):
    image: str  # Base64 encoded JPEG/PNG frame
    model_type: str = "vit"  # "vit" or "custom"
    scale_factor: float = 1.1
    min_neighbors: int = 5

@app.post("/inference")
async def run_inference(data: FrameData):
    """
    Decodes real-time base64 frame, detects face with dynamic calibration, 
    runs classification on either HuggingFace ViT or custom local CNN model, 
    maps to Action Units, and returns telemetry JSON including latencies and grayscale crop.
    """
    global emotion_classifier, local_cnn_model, device
    if emotion_classifier is None or local_cnn_model is None:
        load_pipeline_model()
        if emotion_classifier is None and local_cnn_model is None:
            return {"face_detected": False, "error": "Models failed to load on backend."}

    try:
        t_start = time.time()
        
        # Decode base64 frame
        encoded_data = data.image.split(",")[1] if "," in data.image else data.image
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"face_detected": False, "error": "Failed to decode image frame."}
            
        # Detect faces with user calibration
        faces = detect_faces(frame, scale_factor=data.scale_factor, min_neighbors=data.min_neighbors)
        if len(faces) == 0:
            return {"face_detected": False}
            
        # Extract the primary face (first detected)
        bbox = faces[0]
        x, y, w, h = bbox
        
        # Preprocess face crop
        face_img = preprocess_face(frame, bbox)
        if face_img is None:
            return {"face_detected": False}
            
        # Base64 encode the 48x48 grayscaled cropped image to display in CRT viewfinder
        _, buffer = cv2.imencode('.jpg', face_img)
        cropped_face_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        
        # Run classification based on chosen model
        distribution = {}
        predicted_emotion = "Neutral"
        confidence = 0.0
        
        inference_start = time.time()
        
        if data.model_type.lower() == "custom" and local_cnn_model is not None:
            # Preprocess crop specifically for PyTorch CNN
            # EmotionCNN takes [B x 1 x 48 x 48] normalized tensor in [0, 1]
            img_tensor = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            
            with torch.no_grad():
                logits = local_cnn_model(img_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                
            # Map probabilities to standard emotions
            for idx, emotion in enumerate(EMOTIONS):
                distribution[emotion] = float(probs[idx])
                
            max_idx = np.argmax(probs)
            predicted_emotion = EMOTIONS[max_idx]
            confidence = float(probs[max_idx])
        else:
            # Hugging Face ViT model
            # Convert BGR/Grayscale crop to PIL Image for Vision Transformer
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB))
            
            # Run classification pipeline
            predictions = emotion_classifier(pil_img)
            
            # Build probability distribution map
            for pred in predictions:
                raw_label = pred["label"]
                score = float(pred["score"])
                standard_label = LABEL_MAPPING.get(raw_label.lower(), "Neutral")
                distribution[standard_label] = score
                
            # Ensure all standard emotions are represented
            for emotion in EMOTIONS:
                if emotion not in distribution:
                    distribution[emotion] = 0.0
                    
            # Highest probability prediction is first in list
            best_pred = predictions[0]
            predicted_emotion = LABEL_MAPPING.get(best_pred["label"].lower(), "Neutral")
            confidence = float(best_pred["score"])
            
        inference_latency = (time.time() - inference_start) * 1000
        total_latency = (time.time() - t_start) * 1000
        
        # Map to Action Units
        au_mapping = map_emotion_to_action_units(predicted_emotion)
        
        return {
            "face_detected": True,
            "bbox": [x, y, w, h],
            "emotion": predicted_emotion,
            "confidence": confidence,
            "distribution": distribution,
            "action_units": au_mapping,
            "inference_latency": inference_latency,
            "total_latency": total_latency,
            "cropped_face": cropped_face_b64
        }
        
    except Exception as e:
        return {"face_detected": False, "error": str(e)}

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEURAL NEXUS // SKEUOMORPHIC CYBERPUNK FER CONSOLE</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Share+Tech+Mono&family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Default: Stealth Cyan Theme */
            --bg-carbon: #030406;
            --panel-steel: #0d1117;
            --panel-border: #1f2937;
            --accent-glow: rgba(0, 240, 255, 0.25);
            --accent-neon: #00F0FF;
            --accent-bright: #38bdf8;
            --metal-bevel: linear-gradient(135deg, #374151 0%, #111827 50%, #030712 100%);
            --steel-profile: linear-gradient(to right, #4b5563, #1f2937, #111827, #1f2937, #4b5563);
            --brass-rivet: radial-gradient(circle at 35% 35%, #eab308, #854d0e);
            --text-neon: #e2e8f0;
            --text-dim: #9ca3af;
            --led-off: #1e1b4b;
        }

        /* Alternate Theme Classes */
        .theme-amber {
            --accent-glow: rgba(245, 158, 11, 0.25);
            --accent-neon: #F59E0B;
            --accent-bright: #FBBF24;
            --text-neon: #fef3c7;
        }
        .theme-green {
            --accent-glow: rgba(34, 197, 94, 0.25);
            --accent-neon: #22C55E;
            --accent-bright: #4ade80;
            --text-neon: #dcfce7;
        }
        .theme-crimson {
            --accent-glow: rgba(239, 68, 68, 0.25);
            --accent-neon: #EF4444;
            --accent-bright: #f87171;
            --text-neon: #fee2e2;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Space Grotesk', sans-serif;
        }

        body {
            background-color: var(--bg-carbon);
            background-image: 
                radial-gradient(circle at 50% 50%, rgba(13, 17, 23, 0.7) 0%, var(--bg-carbon) 100%),
                repeating-linear-gradient(0deg, rgba(0, 0, 0, 0.15) 0px, rgba(0, 0, 0, 0.15) 1px, transparent 1px, transparent 2px);
            color: var(--text-neon);
            min-height: 100vh;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            position: relative;
        }

        /* Ambient scanline overlay */
        body::before {
            content: " ";
            display: block;
            position: fixed;
            top: 0; left: 0; bottom: 0; right: 0;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.04), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.04));
            z-index: 999;
            background-size: 100% 4px, 6px 100%;
            pointer-events: none;
            opacity: 0.35;
        }

        /* 3D Skeuomorphic Header Console */
        header {
            width: 100%;
            padding: 1.2rem 2.5rem;
            background: #111827;
            background-image: var(--metal-bevel);
            border-bottom: 4px solid #1f2937;
            box-shadow: inset 0 2px 0 rgba(255,255,255,0.05), 0 8px 24px rgba(0,0,0,0.9);
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .steel-bezel {
            border: 3px solid;
            border-image: var(--steel-profile) 1;
            padding: 0.5rem 1.8rem;
            background: #030712;
            box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.9), 0 0 10px var(--accent-glow);
            position: relative;
        }

        .logo {
            font-weight: 800;
            font-size: 1.5rem;
            letter-spacing: 4px;
            color: var(--accent-neon);
            text-shadow: 0 0 12px var(--accent-glow);
            font-family: 'Share Tech Mono', monospace;
        }

        .logo span {
            color: #ffffff;
            font-weight: 300;
        }

        .status-badge {
            background: #030712;
            border: 2px solid var(--panel-border);
            box-shadow: inset 0 0 12px rgba(0,0,0,0.9);
            color: var(--text-neon);
            padding: 0.5rem 1.5rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 0.9rem;
            font-family: 'Share Tech Mono', monospace;
            letter-spacing: 1.5px;
        }

        /* Mechanical LED Indicator */
        .status-led {
            width: 12px;
            height: 12px;
            background-color: var(--led-off);
            border-radius: 50%;
            box-shadow: 0 0 2px rgba(0,0,0,0.5);
            transition: all 0.3s ease;
            border: 1px solid #111827;
        }

        .status-led.active {
            background-color: var(--accent-neon);
            box-shadow: 0 0 14px var(--accent-neon), inset 0 2px 2px white;
        }

        /* 3D Dashboard Main Container Grid */
        .container {
            flex: 1;
            max-width: 1440px;
            margin: 0 auto;
            width: 100%;
            padding: 2rem;
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 2rem;
        }

        /* Skeuomorphic Panel Base */
        .panel {
            background: var(--panel-steel);
            background-image: linear-gradient(135deg, #182232 0%, #0d131f 100%);
            border: 3px solid var(--panel-border);
            box-shadow: 
                inset 0 2px 2px rgba(255,255,255,0.05),
                3px 3px 0px rgba(0,0,0,0.5),
                8px 8px 25px rgba(0,0,0,0.9);
            border-radius: 14px;
            padding: 1.8rem;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
        }

        /* Rivets */
        .rivet {
            position: absolute;
            width: 10px;
            height: 10px;
            background: var(--brass-rivet);
            border-radius: 50%;
            border: 1px solid #030712;
            box-shadow: 1px 1px 2px rgba(0,0,0,0.7), inset -1px -1px 1px rgba(255,255,255,0.1);
        }
        .rivet-tl { top: 8px; left: 8px; }
        .rivet-tr { top: 8px; right: 8px; }
        .rivet-bl { bottom: 8px; left: 8px; }
        .rivet-br { bottom: 8px; right: 8px; }

        .panel-title {
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--accent-neon);
            margin-bottom: 0.5rem;
            border-bottom: 2px solid var(--panel-border);
            padding-bottom: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: 'Share Tech Mono', monospace;
        }

        /* Main Viewport Frame (Webcam Canvas) */
        .viewport-wrapper {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }

        .viewfinder-bezel {
            border: 10px solid #030712;
            background: #000;
            box-shadow: 
                inset 0 0 40px rgba(0,0,0,1), 
                0 0 0 2px var(--panel-border),
                0 8px 20px rgba(0,0,0,0.9);
            border-radius: 10px;
            position: relative;
            width: 100%;
            aspect-ratio: 4/3;
            overflow: hidden;
        }

        video, canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        /* CRT Viewport Overlay scanline sweeps */
        .laser-sweep {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(to bottom, transparent, var(--accent-neon), transparent);
            box-shadow: 0 0 15px var(--accent-neon);
            opacity: 0.6;
            pointer-events: none;
            animation: laser-sweep-anim 4s infinite linear;
        }

        @keyframes laser-sweep-anim {
            0% { top: 0%; }
            100% { top: 100%; }
        }

        /* 3D Model comparative rocker Switch */
        .rocker-container {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            background: #030712;
            padding: 1rem;
            border-radius: 8px;
            border: 2px solid var(--panel-border);
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.8);
            margin-bottom: 0.5rem;
        }

        .rocker-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-dim);
            font-family: 'Share Tech Mono', monospace;
        }

        .rocker-switch {
            position: relative;
            width: 100%;
            height: 42px;
            background-color: #111827;
            border-radius: 6px;
            border: 2px solid #374151;
            box-shadow: inset 0 3px 6px rgba(0,0,0,0.9);
            display: flex;
            overflow: hidden;
            cursor: pointer;
        }

        .rocker-btn {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 1px;
            z-index: 2;
            color: var(--text-dim);
            font-family: 'Share Tech Mono', monospace;
            text-transform: uppercase;
            transition: all 0.2s ease;
        }

        .rocker-btn.active {
            color: #ffffff;
            text-shadow: 0 0 8px var(--accent-neon);
        }

        .rocker-slider {
            position: absolute;
            top: 2px;
            left: 2px;
            width: calc(50% - 2px);
            height: calc(100% - 4px);
            background: linear-gradient(to bottom, #4b5563 0%, #1f2937 100%);
            border: 2px solid var(--accent-neon);
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.8), inset 0 1px 0 rgba(255,255,255,0.1);
            transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 1;
        }

        /* 3D Knobs / Theme selector dials */
        .knob-panel {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .knob-capsule {
            background: #030712;
            border: 2px solid var(--panel-border);
            border-radius: 8px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.8rem;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.8);
            position: relative;
        }

        .knob-title {
            font-size: 0.7rem;
            font-family: 'Share Tech Mono', monospace;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            text-align: center;
        }

        .knob-housing {
            position: relative;
            width: 70px;
            height: 70px;
            border-radius: 50%;
            background: radial-gradient(circle, #374151, #0f172a);
            border: 4px solid #1e293b;
            box-shadow: 0 4px 6px rgba(0,0,0,0.6), inset 0 2px 3px rgba(255,255,255,0.1);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.1s ease;
        }

        .knob-marker {
            position: absolute;
            top: 6px;
            width: 4px;
            height: 18px;
            background-color: var(--accent-neon);
            box-shadow: 0 0 6px var(--accent-neon);
            border-radius: 2px;
            transform-origin: bottom center;
            transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Preprocessor CRT Viewfinder bezel */
        .crt-bezel {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            border: 6px solid #030712;
            background: #000;
            box-shadow: inset 0 0 15px rgba(0,240,255,0.4), 0 4px 8px rgba(0,0,0,0.8), 0 0 0 2px var(--panel-border);
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .crt-bezel canvas {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            opacity: 0.85;
            filter: contrast(1.4) brightness(1.2) sepia(0.3);
        }

        .crt-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: radial-gradient(rgba(0,240,255,0.15) 60%, rgba(0,0,0,0.8) 100%), linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.4) 50%);
            background-size: 100% 100%, 100% 4px;
            pointer-events: none;
            z-index: 10;
        }

        /* Slide Calibration Drawer Panel */
        .calibration-drawer {
            background: #030712;
            border: 2px solid var(--panel-border);
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.8);
            border-radius: 8px;
            padding: 1.2rem;
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
        }

        .slider-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .slider-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            font-family: 'Share Tech Mono', monospace;
            text-transform: uppercase;
            color: var(--text-dim);
            letter-spacing: 1px;
        }

        .slider-val {
            color: var(--accent-neon);
            font-weight: 700;
        }

        .range-slider {
            -webkit-appearance: none;
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: #1f2937;
            outline: none;
            border: 1px solid #374151;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.8);
        }

        .range-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 4px;
            background: linear-gradient(to bottom, #4b5563 0%, #1f2937 100%);
            border: 2px solid var(--accent-neon);
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.6);
        }

        /* Biometric EKG telemetry panel */
        .bio-viewport {
            background: #030712;
            border: 2px solid var(--panel-border);
            box-shadow: inset 0 0 18px rgba(0,240,255,0.15), inset 0 2px 10px rgba(0,0,0,0.9);
            border-radius: 8px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
            position: relative;
        }

        .ecg-grid {
            width: 100%;
            height: 100px;
            background: #000000;
            border-radius: 4px;
            border: 1px solid var(--panel-border);
            position: relative;
            overflow: hidden;
        }

        .ecg-canvas {
            width: 100%;
            height: 100%;
        }

        .bio-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .bio-card {
            background: #111827;
            border: 1px solid var(--panel-border);
            border-radius: 4px;
            padding: 0.6rem;
            text-align: center;
        }

        .bio-lbl {
            font-size: 0.65rem;
            font-family: 'Share Tech Mono', monospace;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .bio-val {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--accent-neon);
            font-family: 'Share Tech Mono', monospace;
            margin-top: 0.2rem;
        }

        /* Session log Cartridge bay */
        .cartridge-bay {
            background: #030712;
            border: 2px solid var(--panel-border);
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.8);
            border-radius: 8px;
            padding: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }

        .cartridge-chassis {
            flex: 1;
            height: 38px;
            background: linear-gradient(180deg, #1f2937 0%, #0d1117 100%);
            border: 2px solid #374151;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.9);
            border-radius: 4px;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .cartridge-insertion {
            position: absolute;
            left: 6px; right: 6px; height: 18px;
            background: repeating-linear-gradient(45deg, #111, #111 4px, #222 4px, #222 8px);
            border: 1px solid #000;
            border-radius: 2px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            transform: translateY(0);
        }

        .cartridge-insertion.mounted {
            transform: translateY(-8px);
            background: var(--accent-neon);
            box-shadow: 0 0 10px var(--accent-glow);
            border-color: #ffffff;
        }

        .cartridge-insertion.mounted::after {
            content: "CARTRIDGE ACTIVE";
            position: absolute;
            top: 2px; left: 0; width: 100%;
            text-align: center;
            font-size: 0.55rem;
            font-family: 'Share Tech Mono', monospace;
            color: #000000;
            font-weight: 900;
            letter-spacing: 1px;
        }

        /* 3D Beveled active buttons and sliders */
        .controls {
            display: flex;
            gap: 1.2rem;
            background: #030712;
            padding: 1.2rem;
            border-radius: 8px;
            border: 2px solid var(--panel-border);
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.8);
        }

        .btn {
            background: linear-gradient(to bottom, #1f2937 0%, #0d1117 100%);
            border: 2px solid var(--panel-border);
            color: var(--text-neon);
            padding: 0.8rem 1.6rem;
            border-radius: 6px;
            font-weight: 700;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            transition: all 0.1s ease;
            box-shadow: 0 4px 0px #030712, 0 6px 12px rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            position: relative;
            flex: 1;
            font-family: 'Share Tech Mono', monospace;
        }

        .btn:hover {
            border-color: var(--accent-neon);
            box-shadow: 0 4px 0px #030712, 0 0px 10px var(--accent-glow);
        }

        .btn:active {
            transform: translateY(4px);
            box-shadow: 0 0px 0px transparent, inset 0 3px 5px rgba(0,0,0,0.9);
        }

        .btn-primary {
            background: linear-gradient(to bottom, #0369a1 0%, #075985 100%);
            color: #ffffff;
            border-color: #0284c7;
        }

        .btn-primary:hover {
            border-color: var(--accent-neon);
        }

        .btn-primary.active {
            background: linear-gradient(to bottom, #b91c1c 0%, #7f1d1d 100%);
            border-color: #ef4444;
        }

        /* Benchmarking Telemetry Display grid */
        .bench-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            background: #030712;
            border: 2px solid var(--panel-border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.8);
        }

        .bench-item {
            background: #111827;
            border: 1px solid var(--panel-border);
            border-radius: 4px;
            padding: 0.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .bench-lbl {
            font-size: 0.65rem;
            font-family: 'Share Tech Mono', monospace;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .bench-val {
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--accent-neon);
            font-family: 'Share Tech Mono', monospace;
        }

        /* Vacuum tube main capsule predictions */
        .display-capsule {
            background: #000;
            border: 3px solid var(--panel-border);
            box-shadow: inset 0 0 25px rgba(0,0,0,0.9), 0 0 15px var(--accent-glow);
            border-radius: 12px;
            padding: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
            border-left: 5px solid var(--accent-neon);
        }

        .display-capsule::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 50%;
            background: linear-gradient(rgba(255,255,255,0.06), transparent);
            pointer-events: none;
        }

        .capsule-label {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-dim);
            letter-spacing: 2px;
            text-transform: uppercase;
            position: absolute;
            top: 6px;
            left: 15px;
        }

        .emotion-main {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--accent-neon);
            letter-spacing: 2px;
            text-shadow: 0 0 15px var(--accent-glow);
            text-transform: uppercase;
            margin-top: 0.5rem;
            font-family: 'Share Tech Mono', monospace;
        }

        .confidence-pill {
            background: #111827;
            border: 2px solid var(--panel-border);
            box-shadow: inset 0 0 8px rgba(0,0,0,0.8);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-weight: bold;
            font-size: 1.2rem;
            font-family: 'Share Tech Mono', monospace;
            color: var(--accent-neon);
            text-shadow: 0 0 8px var(--accent-glow);
        }

        /* 3D Equalizer progress bar panel */
        .equalizer-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            background: #030712;
            padding: 1.5rem;
            border-radius: 8px;
            border: 2px solid var(--panel-border);
            box-shadow: inset 0 5px 15px rgba(0,0,0,0.9);
        }

        .eq-row {
            display: grid;
            grid-template-columns: 100px 1fr 50px;
            align-items: center;
            gap: 1.2rem;
        }

        .eq-label {
            font-size: 0.8rem;
            font-weight: bold;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'Share Tech Mono', monospace;
        }

        .led-grid-bg {
            background: #111827;
            height: 16px;
            border-radius: 3px;
            overflow: hidden;
            border: 1px solid #374151;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.9);
            position: relative;
        }

        .led-fill {
            height: 100%;
            background-image: linear-gradient(to right, 
                var(--led-off) 0%, var(--accent-neon) 60%, var(--accent-bright) 100%
            );
            background-size: 100% 100%;
            width: 0%;
            transition: width 0.15s ease-out;
            box-shadow: 0 0 10px var(--accent-glow);
            position: relative;
        }

        .led-fill::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: repeating-linear-gradient(90deg, 
                transparent, transparent 6px, 
                #030712 6px, #030712 8px
            );
            pointer-events: none;
        }

        .eq-val {
            font-size: 0.85rem;
            font-family: 'Share Tech Mono', monospace;
            font-weight: bold;
            color: var(--accent-neon);
            text-align: right;
        }

        .au-panel {
            background: #030712;
            border: 2px solid var(--panel-border);
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: inset 0 5px 15px rgba(0,0,0,0.9);
        }

        .au-title-sub {
            font-size: 0.75rem;
            font-weight: bold;
            color: var(--text-dim);
            letter-spacing: 2px;
            margin-bottom: 0.8rem;
            text-transform: uppercase;
            font-family: 'Share Tech Mono', monospace;
        }

        .au-desc {
            font-size: 0.95rem;
            color: var(--accent-neon);
            margin-bottom: 1rem;
            font-weight: 600;
            font-family: 'Share Tech Mono', monospace;
        }

        .au-list {
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }

        .au-pill {
            background: linear-gradient(to bottom, #1f2937 0%, #0d1117 100%);
            border: 1px solid var(--panel-border);
            box-shadow: 1px 1px 2px rgba(0,0,0,0.8);
            color: var(--text-neon);
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
            font-family: 'Share Tech Mono', monospace;
        }

        footer {
            text-align: center;
            padding: 1.8rem;
            color: var(--text-dim);
            font-size: 0.8rem;
            font-family: 'Share Tech Mono', monospace;
            background: #030712;
            border-top: 4px solid var(--panel-border);
            margin-top: auto;
            letter-spacing: 2px;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.9);
        }
    </style>
</head>
<body>

    <header>
        <div class="steel-bezel">
            <div class="logo">NEURAL NEXUS // <span>FER CONSOLE</span></div>
        </div>
        <div class="status-badge">
            <span class="status-led" id="mainLed"></span>
            <span id="backendStatus">CONSOLE OFFLINE</span>
        </div>
    </header>

    <div class="container">
        <!-- Column 1: Viewport & Calibration controls -->
        <div class="panel">
            <span class="rivet rivet-tl"></span>
            <span class="rivet rivet-tr"></span>
            <span class="rivet rivet-bl"></span>
            <span class="rivet rivet-br"></span>
            
            <div class="panel-title">
                TACTILE VIEWPORT INGESTION
                <span id="fpsTelemetry" style="color: var(--text-dim); font-size: 0.75rem;">FPS: 0.0 (OFFLINE)</span>
            </div>
            
            <div class="viewport-wrapper">
                <div class="viewfinder-bezel">
                    <video id="webcam" autoplay playsinline muted></video>
                    <canvas id="overlay"></canvas>
                    <div class="laser-sweep"></div>
                </div>
            </div>

            <!-- Sound & Theme knobs dashboard drawer -->
            <div class="knob-panel">
                <!-- Theme Dial Knob -->
                <div class="knob-capsule">
                    <span class="knob-title">Palette Selector</span>
                    <div class="knob-housing" id="themeKnob" data-angle="0">
                        <div class="knob-marker" id="themeKnobMarker"></div>
                    </div>
                    <span id="themeLabel" style="font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; color: var(--accent-neon);">CYAN</span>
                </div>

                <!-- Sonification Volume / Engage Knob -->
                <div class="knob-capsule">
                    <span class="knob-title">Sonic Deck Engaged</span>
                    <div class="knob-housing" id="soundKnob" data-angle="-135">
                        <div class="knob-marker" id="soundKnobMarker" style="transform: rotate(-135deg);"></div>
                    </div>
                    <span id="soundLabel" style="font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; color: var(--text-dim);">AUDIO MUTED</span>
                </div>
            </div>

            <!-- Calibration controls slider drawers -->
            <div class="calibration-drawer">
                <div class="panel-title" style="border-bottom: 1px solid #1f2937; margin-bottom: 0; padding-bottom: 0.4rem; font-size: 0.9rem;">
                    Hardware Calibration board
                </div>
                
                <div style="display: flex; gap: 1rem; align-items: center;">
                    <!-- Circular CRT preprocessor viewfinder -->
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 0.4rem;">
                        <div class="crt-bezel">
                            <canvas id="crtCanvas" width="48" height="48"></canvas>
                            <div class="crt-overlay"></div>
                        </div>
                        <span style="font-size: 0.6rem; font-family: 'Share Tech Mono', monospace; color: var(--text-dim); text-transform: uppercase;">CRT Feed</span>
                    </div>

                    <div style="flex: 1; display: flex; flex-direction: column; gap: 1rem;">
                        <!-- Scale Factor Slider -->
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>Cascade Scale Factor</span>
                                <span class="slider-val" id="scaleVal">1.10</span>
                            </div>
                            <input type="range" class="range-slider" id="scaleSlider" min="1.05" max="1.4" step="0.01" value="1.10">
                        </div>

                        <!-- Min Neighbors Slider -->
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>Min Target Neighbors</span>
                                <span class="slider-val" id="neighborsVal">5</span>
                            </div>
                            <input type="range" class="range-slider" id="neighborsSlider" min="2" max="10" step="1" value="5">
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" id="toggleStream">Engage console</button>
                <button class="btn" id="toggleDraw" style="border-color: var(--accent-neon); color: #ffffff; text-shadow: 0 0 6px var(--accent-glow);">Draw reticles</button>
            </div>
        </div>

        <!-- Column 2: Comparative Engine & Telemetry -->
        <div class="analytics">
            <!-- Rocket Engine Switch -->
            <div class="panel">
                <span class="rivet rivet-tl"></span>
                <span class="rivet rivet-tr"></span>
                <span class="rivet rivet-bl"></span>
                <span class="rivet rivet-br"></span>
                
                <div class="panel-title">Model Decision Matrix</div>

                <div class="rocker-container">
                    <span class="rocker-label">Neural Engine Select</span>
                    <div class="rocker-switch" id="modelRocker">
                        <div class="rocker-slider" id="rockerSlider"></div>
                        <div class="rocker-btn active" id="rockerVit" data-model="vit">ViT Model</div>
                        <div class="rocker-btn" id="rockerCnn" data-model="custom">Custom CNN</div>
                    </div>
                </div>

                <!-- Session Cartridge Floppy slots -->
                <div class="cartridge-bay">
                    <div style="display: flex; flex-direction: column; gap: 0.1rem;">
                        <span style="font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; font-weight: bold; color: var(--text-neon);">MEMORY CHASSIS</span>
                        <span style="font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: var(--text-dim);">SLOT 01 // TELEMETRY LOG</span>
                    </div>
                    <div class="cartridge-chassis">
                        <div class="cartridge-insertion" id="cartridgeSlot"></div>
                    </div>
                    <button class="btn" id="btnCartridge" style="padding: 0.4rem 0.8rem; font-size: 0.7rem; box-shadow: 0 2px 0px #030712;">Insert Tape</button>
                </div>

                <!-- Benchmarking Metrics -->
                <div class="bench-grid">
                    <div class="bench-item">
                        <span class="bench-lbl">Inference Latency</span>
                        <span class="bench-val" id="infLatency">0.00 ms</span>
                    </div>
                    <div class="bench-item">
                        <span class="bench-lbl">Model Weight Size</span>
                        <span class="bench-val" id="modelWeight">~343 MB</span>
                    </div>
                    <div class="bench-item">
                        <span class="bench-lbl">Total E2E Latency</span>
                        <span class="bench-val" id="totalLatency">0.00 ms</span>
                    </div>
                    <div class="bench-item">
                        <span class="bench-lbl">Processor Pipeline</span>
                        <span class="bench-val" id="processorType">CPU</span>
                    </div>
                </div>
            </div>

            <!-- Telemetry diagnostics -->
            <div class="panel">
                <span class="rivet rivet-tl"></span>
                <span class="rivet rivet-tr"></span>
                <span class="rivet rivet-bl"></span>
                <span class="rivet rivet-br"></span>
                
                <div class="panel-title">Active Diagnostics</div>
                
                <div class="display-capsule">
                    <span class="capsule-label">Classified State</span>
                    <span id="emotion" class="emotion-main">OFFLINE</span>
                    <span id="confidence" class="confidence-pill">0.0%</span>
                </div>

                <!-- Simulated EKG wave canvas -->
                <div class="bio-viewport">
                    <span class="capsule-label" style="top: 4px; left: 8px;">Simulated Biometrics telemetry</span>
                    <div class="ecg-grid">
                        <canvas class="ecg-canvas" id="ecgCanvas"></canvas>
                    </div>
                    <div class="bio-stats">
                        <div class="bio-card">
                            <span class="bio-lbl">Heart Rate</span>
                            <span class="bio-val" id="bioBpm">72 BPM</span>
                        </div>
                        <div class="bio-card">
                            <span class="bio-lbl">Stress Index</span>
                            <span class="bio-val" id="bioStress">12%</span>
                        </div>
                    </div>
                </div>

                <!-- Skeuomorphic Equalizer Bars -->
                <div class="equalizer-container" id="equalizerContainer">
                    <!-- Dynamic Bars -->
                </div>

                <!-- FACS details -->
                <div class="au-panel">
                    <div class="au-title-sub">FACIAL ACTION SYSTEM (FACS) METRIC</div>
                    <div id="auDesc" class="au-desc">System idle. Engage stream viewport to acquire targets.</div>
                    <ul id="auList" class="au-list">
                        <!-- Dynamic AUs -->
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <footer>
        STEALTH CONSOLE // INTEGRATED MILITARY-GRADE EMOTION TELEMETRY ENGINE // 2026-V4
    </footer>

    <script>
        // DOM Bindings
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        const toggleBtn = document.getElementById('toggleStream');
        const toggleDrawBtn = document.getElementById('toggleDraw');
        const mainLed = document.getElementById('mainLed');
        const backendStatus = document.getElementById('backendStatus');
        
        // Calibration Controls
        const scaleSlider = document.getElementById('scaleSlider');
        const scaleVal = document.getElementById('scaleVal');
        const neighborsSlider = document.getElementById('neighborsSlider');
        const neighborsVal = document.getElementById('neighborsVal');
        
        // Dynamic Viewfinders
        const crtCanvas = document.getElementById('crtCanvas');
        const crtCtx = crtCanvas.getContext('2d');
        const fpsTelemetry = document.getElementById('fpsTelemetry');
        
        // Model Selection rocker
        const modelRocker = document.getElementById('modelRocker');
        const rockerSlider = document.getElementById('rockerSlider');
        const rockerVit = document.getElementById('rockerVit');
        const rockerCnn = document.getElementById('rockerCnn');
        
        // Telemetry bench elements
        const infLatency = document.getElementById('infLatency');
        const totalLatency = document.getElementById('totalLatency');
        const modelWeight = document.getElementById('modelWeight');
        const processorType = document.getElementById('processorType');
        
        // Session Floppy elements
        const btnCartridge = document.getElementById('btnCartridge');
        const cartridgeSlot = document.getElementById('cartridgeSlot');
        
        // Theme knob dial
        const themeKnob = document.getElementById('themeKnob');
        const themeKnobMarker = document.getElementById('themeKnobMarker');
        const themeLabel = document.getElementById('themeLabel');
        
        // Audio knob dial
        const soundKnob = document.getElementById('soundKnob');
        const soundKnobMarker = document.getElementById('soundKnobMarker');
        const soundLabel = document.getElementById('soundLabel');
        
        // EKG canvas & bio telemetry
        const ecgCanvas = document.getElementById('ecgCanvas');
        const ecgCtx = ecgCanvas.getContext('2d');
        const bioBpm = document.getElementById('bioBpm');
        const bioStress = document.getElementById('bioStress');
        
        // Runtime variables
        let streaming = false;
        let drawBBox = true;
        let streamInterval = null;
        let currentModel = "vit";
        let activeTheme = "cyan"; // "cyan", "amber", "green", "crimson"
        let soundEngaged = false;
        let volumeLevel = 0.0;
        
        // Cartridge Log telemetry
        let sessionTapeInserted = false;
        let telemetryLog = [];
        
        // Sound deck context
        let audioCtx = null;
        let synthOsc = null;
        let synthGain = null;
        let targetSpeechSpeechTime = 0;
        
        const emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"];
        
        // Set colors matching selected neon themes
        const themeHexColors = {
            cyan: "#00F0FF",
            amber: "#F59E0B",
            green: "#22C55E",
            crimson: "#EF4444"
        };
        
        // Equalizer setup
        const equalizerContainer = document.getElementById('equalizerContainer');
        const barElements = {};
        const valElements = {};
        
        emotions.forEach(emotion => {
            const row = document.createElement('div');
            row.className = 'eq-row';
            
            const label = document.createElement('span');
            label.className = 'eq-label';
            label.textContent = emotion;
            
            const ledBg = document.createElement('div');
            ledBg.className = 'led-grid-bg';
            
            const ledFill = document.createElement('div');
            ledFill.className = 'led-fill';
            
            const val = document.createElement('span');
            val.className = 'eq-val';
            val.textContent = '0.0%';
            
            ledBg.appendChild(ledFill);
            row.appendChild(label);
            row.appendChild(ledBg);
            row.appendChild(val);
            equalizerContainer.appendChild(row);
            
            barElements[emotion] = ledFill;
            valElements[emotion] = val;
        });

        // 3D Rocker switch model selection
        rockerVit.addEventListener('click', () => selectModel("vit"));
        rockerCnn.addEventListener('click', () => selectModel("custom"));
        
        function selectModel(model) {
            currentModel = model;
            if (model === "vit") {
                rockerSlider.style.transform = "translateX(0%)";
                rockerVit.classList.add("active");
                rockerCnn.classList.remove("active");
                modelWeight.textContent = "~343 MB";
                speakVoice("Neural engine set to Vision Transformer model");
            } else {
                rockerSlider.style.transform = "translateX(100%)";
                rockerVit.classList.remove("active");
                rockerCnn.classList.add("active");
                modelWeight.textContent = "10.6 MB";
                speakVoice("Neural engine set to local Custom Convolutional Network");
            }
        }
        
        // Slider calibration listeners
        scaleSlider.addEventListener('input', (e) => {
            scaleVal.textContent = parseFloat(e.target.value).toFixed(2);
        });
        neighborsSlider.addEventListener('input', (e) => {
            neighborsVal.textContent = parseInt(e.target.value);
        });
        
        // Dial 1 Knob theme palette listener
        const themesList = ["cyan", "amber", "green", "crimson"];
        themeKnob.addEventListener('click', () => {
            let currentAngle = parseInt(themeKnob.dataset.angle) || 0;
            currentAngle += 90;
            if (currentAngle >= 360) currentAngle = 0;
            
            themeKnob.dataset.angle = currentAngle;
            themeKnobMarker.style.transform = `rotate(${currentAngle}deg)`;
            
            // Toggle body class
            document.body.classList.remove('theme-amber', 'theme-green', 'theme-crimson');
            
            const index = Math.floor(currentAngle / 90);
            activeTheme = themesList[index];
            themeLabel.textContent = activeTheme.toUpperCase();
            
            if (activeTheme !== "cyan") {
                document.body.classList.add(`theme-${activeTheme}`);
            }
            
            // Tone beep feedback
            playTone(440 + currentAngle, 0.05, "sine");
        });
        
        // Dial 2 sound deck controller dial listener
        soundKnob.addEventListener('click', () => {
            let currentAngle = parseInt(soundKnob.dataset.angle) || -135;
            // Dials range from -135deg (off/muted) to 135deg (max volume)
            currentAngle += 45;
            if (currentAngle > 135) currentAngle = -135;
            
            soundKnob.dataset.angle = currentAngle;
            soundKnobMarker.style.transform = `rotate(${currentAngle}deg)`;
            
            if (currentAngle === -135) {
                soundEngaged = false;
                soundLabel.textContent = "AUDIO MUTED";
                soundLabel.style.color = "var(--text-dim)";
                stopAudioHum();
            } else {
                soundEngaged = true;
                volumeLevel = (currentAngle + 135) / 270.0; // scale volume linearly
                soundLabel.textContent = `VOLUME: ${Math.round(volumeLevel * 100)}%`;
                soundLabel.style.color = "var(--accent-neon)";
                
                initAudioHum();
                if (synthGain) {
                    synthGain.gain.setValueAtTime(volumeLevel * 0.1, audioCtx.currentTime);
                }
                playTone(600 + (volumeLevel * 400), 0.08, "triangle");
            }
        });

        // Cartridge Deck inserting logs tape
        btnCartridge.addEventListener('click', () => {
            if (!sessionTapeInserted) {
                sessionTapeInserted = true;
                cartridgeSlot.classList.add("mounted");
                btnCartridge.textContent = "EJECT TAPE";
                telemetryLog = [];
                speakVoice("Telemetry magnetic cartridge successfully mounted. Session data logging activated.");
                playTone(300, 0.1, "sawtooth");
                setTimeout(() => playTone(500, 0.1, "sawtooth"), 100);
            } else {
                // Eject logs and download
                sessionTapeInserted = false;
                cartridgeSlot.classList.remove("mounted");
                btnCartridge.textContent = "INSERT TAPE";
                
                speakVoice("Cartridge ejected. Transmitting saved diagnostics.");
                playTone(500, 0.1, "sawtooth");
                setTimeout(() => playTone(300, 0.1, "sawtooth"), 100);
                
                if (telemetryLog.length > 0) {
                    downloadLogs();
                } else {
                    alert("Telemetry cartridge ejected empty. Engage camera capture first to log session tracks.");
                }
            }
        });
        
        function downloadLogs() {
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(telemetryLog, null, 2));
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute("href", dataStr);
            downloadAnchor.setAttribute("download", `fer_session_log_${Date.now()}.json`);
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        }

        // Web Audio API Synth Hum Routines
        function initAudioHum() {
            if (audioCtx) return;
            try {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                
                // Deep Cyberpunk Reactor hum oscillator setup
                synthOsc = audioCtx.createOscillator();
                synthOsc.type = "sawtooth";
                synthOsc.frequency.setValueAtTime(45, audioCtx.currentTime); // Deep hum
                
                // Bandpass dynamic filter to replicate server fans hum
                const bandpass = audioCtx.createBiquadFilter();
                bandpass.type = "lowpass";
                bandpass.frequency.setValueAtTime(120, audioCtx.currentTime);
                
                synthGain = audioCtx.createGain();
                synthGain.gain.setValueAtTime(volumeLevel * 0.1, audioCtx.currentTime);
                
                synthOsc.connect(bandpass);
                bandpass.connect(synthGain);
                synthGain.connect(audioCtx.destination);
                
                synthOsc.start();
            } catch (err) {
                console.error("Audio Synthesis initiation failed:", err);
            }
        }
        
        function stopAudioHum() {
            if (synthOsc) {
                try {
                    synthOsc.stop();
                } catch(e){}
                    synthOsc = null;
            }
            audioCtx = null;
        }

        function playTone(freq, duration, type="sine") {
            if (!soundEngaged) return;
            try {
                const ctx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                
                osc.type = type;
                osc.frequency.setValueAtTime(freq, ctx.currentTime);
                
                gain.gain.setValueAtTime(volumeLevel * 0.15, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
                
                osc.connect(gain);
                gain.connect(ctx.destination);
                
                osc.start();
                osc.stop(ctx.currentTime + duration);
            } catch(e){}
        }
        
        function speakVoice(text) {
            if (!soundEngaged) return;
            // Restrict speech synthesis spam rate
            const now = Date.now();
            if (now < targetSpeechSpeechTime) return;
            targetSpeechSpeechTime = now + 4000;
            
            try {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.0;
                utterance.pitch = 0.65; // Robotic synthesizer pitch
                window.speechSynthesis.speak(utterance);
            } catch(e){}
        }

        // Vector EKG Monitor drawing engine
        let ecgSweepX = 0;
        const ecgPoints = [];
        let bioPulseBPM = 72;
        let bioCurrentStress = 12;
        let lastEcgY = 50;

        function animateECG() {
            requestAnimationFrame(animateECG);
            
            const w = ecgCanvas.width = ecgCanvas.clientWidth;
            const h = ecgCanvas.height = ecgCanvas.clientHeight;
            
            ecgCtx.fillStyle = "rgba(0,0,0,0.06)";
            ecgCtx.fillRect(0, 0, w, h);
            
            // Draw fluorescent cyan Grid Background
            ecgCtx.strokeStyle = "rgba(0, 240, 255, 0.05)";
            ecgCtx.lineWidth = 1;
            
            const gridSize = 16;
            for (let x = 0; x < w; x += gridSize) {
                ecgCtx.beginPath();
                ecgCtx.moveTo(x, 0);
                ecgCtx.lineTo(x, h);
                ecgCtx.stroke();
            }
            for (let y = 0; y < h; y += gridSize) {
                ecgCtx.beginPath();
                ecgCtx.moveTo(0, y);
                ecgCtx.lineTo(w, y);
                ecgCtx.stroke();
            }
            
            // Dynamic pulse timing based on active classification BPM
            // 72 BPM is ~1.2Hz. 130 BPM is ~2.1Hz.
            const speed = 2.8;
            ecgSweepX += speed;
            if (ecgSweepX >= w) {
                ecgSweepX = 0;
                ecgCtx.clearRect(0, 0, w, h);
            }
            
            // Calculate height of the peak matching EKG shapes
            let targetY = h / 2;
            const cycleMs = (60 / bioPulseBPM) * 1000;
            const timeInCycle = (Date.now() % cycleMs);
            
            if (streaming) {
                if (timeInCycle < 80) {
                    // P wave (small up)
                    targetY = h / 2 - 4;
                } else if (timeInCycle >= 100 && timeInCycle < 130) {
                    // Q deep drop
                    targetY = h / 2 + 10;
                } else if (timeInCycle >= 130 && timeInCycle < 170) {
                    // R high peak (scale amplitude by stress)
                    const amp = 30 + (bioCurrentStress * 0.35);
                    targetY = h / 2 - amp;
                } else if (timeInCycle >= 170 && timeInCycle < 210) {
                    // S deep valley
                    targetY = h / 2 + 14;
                } else if (timeInCycle >= 250 && timeInCycle < 330) {
                    // T wave
                    targetY = h / 2 - 8;
                }
            } else {
                // Static noise offline signal
                targetY = h / 2 + (Math.random() - 0.5) * 2;
            }
            
            // Draw fluorescent vector glow lines
            ecgCtx.strokeStyle = themeHexColors[activeTheme] || "#00F0FF";
            ecgCtx.shadowBlur = 8;
            ecgCtx.shadowColor = ecgCtx.strokeStyle;
            ecgCtx.lineWidth = 2.5;
            ecgCtx.beginPath();
            ecgCtx.moveTo(ecgSweepX - speed, lastEcgY);
            ecgCtx.lineTo(ecgSweepX, targetY);
            ecgCtx.stroke();
            
            ecgCtx.shadowBlur = 0; // reset glow
            lastEcgY = targetY;
        }
        
        // Start ECG Animation
        animateECG();

        // Viewfinder and bounding controls
        toggleDrawBtn.addEventListener('click', () => {
            drawBBox = !drawBBox;
            toggleDrawBtn.style.background = drawBBox ? 'linear-gradient(to bottom, #1f2937 0%, #0d1117 100%)' : '#030712';
            toggleDrawBtn.style.borderColor = drawBBox ? 'var(--accent-neon)' : 'var(--panel-border)';
            playTone(350, 0.05);
        });

        toggleBtn.addEventListener('click', async () => {
            if (streaming) {
                stopStream();
            } else {
                await startStream();
            }
        });

        async function startStream() {
            try {
                const constraints = {
                    video: { width: 640, height: 480, facingMode: "user" }
                };
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                streaming = true;
                
                mainLed.classList.add("active");
                backendStatus.textContent = "TELEMETRY ACTIVE";
                backendStatus.style.color = "var(--accent-neon)";
                
                toggleBtn.textContent = "STOP STREAM";
                toggleBtn.classList.add("active");
                
                video.addEventListener('loadedmetadata', () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    // Start capture loops
                    streamInterval = setInterval(captureFrame, 180); // fluid 5.5 FPS for real-time telemetry
                });
                
                speakVoice("Camera pipeline active. Initiating computer vision face tracker.");
                playTone(400, 0.1, "sine");
                setTimeout(() => playTone(800, 0.1, "sine"), 80);
            } catch (err) {
                console.error("Camera access failed:", err);
                alert("Camera access denied or failed to bind webcam stream.");
            }
        }

        function stopStream() {
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
            }
            video.srcObject = null;
            streaming = false;
            clearInterval(streamInterval);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Clear CRT crop viewfinder
            crtCtx.clearRect(0, 0, crtCanvas.width, crtCanvas.height);
            
            mainLed.classList.remove("active");
            backendStatus.textContent = "CONSOLE OFFLINE";
            backendStatus.style.color = "var(--text-dim)";
            
            toggleBtn.textContent = "ENGAGE CONSOLE";
            toggleBtn.classList.remove("active");
            
            fpsTelemetry.textContent = "FPS: 0.0 (OFFLINE)";
            infLatency.textContent = "0.00 ms";
            totalLatency.textContent = "0.00 ms";
            processorType.textContent = "CPU";
            
            document.getElementById('emotion').textContent = "OFFLINE";
            document.getElementById('confidence').textContent = "0.0%";
            
            bioBpm.textContent = "72 BPM";
            bioStress.textContent = "12%";
            bioPulseBPM = 72;
            bioCurrentStress = 12;
            
            document.getElementById('auDesc').textContent = "System idle. Engage stream viewport to acquire targets.";
            document.getElementById('auList').innerHTML = '';
            
            emotions.forEach(emotion => {
                barElements[emotion].style.width = '0%';
                valElements[emotion].textContent = '0.0%';
            });
            
            speakVoice("Camera stream offline.");
            playTone(800, 0.1, "sine");
            setTimeout(() => playTone(400, 0.1, "sine"), 80);
        }

        let lastFrameTime = Date.now();

        async function captureFrame() {
            if (!streaming) return;
            
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = 320; // Downscale to maximize transmission bandwidth
            captureCanvas.height = 240;
            const captureCtx = captureCanvas.getContext('2d');
            captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
            
            const base64Image = captureCanvas.toDataURL('image/jpeg', 0.65);
            
            // Grab calibration values
            const sf = parseFloat(scaleSlider.value);
            const mn = parseInt(neighborsSlider.value);
            
            try {
                const reqStart = Date.now();
                const response = await fetch('/inference', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        image: base64Image,
                        model_type: currentModel,
                        scale_factor: sf,
                        min_neighbors: mn
                    })
                });
                
                const result = await response.json();
                
                // Compute visual framerate (FPS)
                const now = Date.now();
                const fps = (1000 / (now - lastFrameTime)).toFixed(1);
                lastFrameTime = now;
                fpsTelemetry.textContent = `FPS: ${fps} (STABLE)`;
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                if (result.face_detected) {
                    // Update CPU benchmarks
                    infLatency.textContent = result.inference_latency.toFixed(2) + " ms";
                    totalLatency.textContent = result.total_latency.toFixed(2) + " ms";
                    processorType.textContent = result.inference_latency < 10.0 ? "CPU [CNN]" : "CPU [ViT]";
                    
                    // Render base64 crop in CRT preprocessor circular monitor
                    if (result.cropped_face) {
                        const img = new Image();
                        img.src = result.cropped_face;
                        img.onload = () => {
                            crtCtx.clearRect(0, 0, crtCanvas.width, crtCanvas.height);
                            crtCtx.drawImage(img, 0, 0, crtCanvas.width, crtCanvas.height);
                            
                            // Draw static raster grid
                            crtCtx.strokeStyle = "rgba(0, 240, 255, 0.15)";
                            crtCtx.lineWidth = 0.5;
                            for (let i = 0; i < crtCanvas.width; i += 4) {
                                crtCtx.beginPath();
                                crtCtx.moveTo(i, 0); crtCtx.lineTo(i, crtCanvas.height);
                                crtCtx.stroke();
                                
                                crtCtx.beginPath();
                                crtCtx.moveTo(0, i); crtCtx.lineTo(crtCanvas.width, i);
                                crtCtx.stroke();
                            }
                        };
                    }
                    
                    // Update active emotion telemetry
                    document.getElementById('emotion').textContent = result.emotion;
                    document.getElementById('confidence').textContent = (result.confidence * 100).toFixed(1) + "%";
                    
                    // Bind simulated biometrics
                    updateBiometrics(result.emotion, result.confidence);
                    
                    // Update Action Units
                    document.getElementById('auDesc').textContent = result.action_units.description;
                    const auList = document.getElementById('auList');
                    auList.innerHTML = '';
                    result.action_units.units.forEach(unit => {
                        const li = document.createElement('li');
                        li.className = 'au-pill';
                        li.textContent = unit;
                        auList.appendChild(li);
                    });
                    
                    // Render segmented equalizer probability bars
                    emotions.forEach(emotion => {
                        const prob = result.distribution[emotion] || 0.0;
                        barElements[emotion].style.width = (prob * 100) + "%";
                        valElements[emotion].textContent = (prob * 100).toFixed(1) + "%";
                    });
                    
                    // Store records in Memory Cartridge Logs if Tape is mounted
                    if (sessionTapeInserted) {
                        telemetryLog.push({
                            timestamp: new Date().toISOString(),
                            model: currentModel,
                            detected_emotion: result.emotion,
                            confidence: result.confidence,
                            ecg_heart_rate: bioPulseBPM,
                            stress_index: bioCurrentStress,
                            inference_latency_ms: result.inference_latency
                        });
                    }
                    
                    // Trigger dynamic target beep pitch on state changes
                    if (result.confidence > 0.6) {
                        playTone(500 + (result.confidence * 200), 0.05, "sine");
                    }
                    
                    // Trigger sound synthesizer status vocal announcements
                    if (result.confidence > 0.85) {
                        speakVoice(`Target localized. State classified as ${result.emotion}. Status secure.`);
                    }
                    
                    // Draw neon canvas corners
                    if (drawBBox) {
                        const [x, y, w, h] = result.bbox;
                        const scaleX = canvas.width / 320;
                        const scaleY = canvas.height / 240;
                        
                        const bx = x * scaleX;
                        const by = y * scaleY;
                        const bw = w * scaleX;
                        const bh = h * scaleY;
                        const len = Math.min(bw, bh) * 0.25;
                        
                        ctx.strokeStyle = themeHexColors[activeTheme] || '#00F0FF';
                        ctx.lineWidth = 3.5;
                        ctx.shadowBlur = 15;
                        ctx.shadowColor = ctx.strokeStyle;
                        
                        ctx.strokeRect(bx, by, bw, bh);
                        
                        // Futuristic Target HUD Brackets
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 4;
                        
                        // Top Left Corner
                        ctx.beginPath();
                        ctx.moveTo(bx + len, by);
                        ctx.lineTo(bx, by);
                        ctx.lineTo(bx, by + len);
                        ctx.stroke();
                        
                        // Top Right Corner
                        ctx.beginPath();
                        ctx.moveTo(bx + bw - len, by);
                        ctx.lineTo(bx + bw, by);
                        ctx.lineTo(bx + bw, by + len);
                        ctx.stroke();
                        
                        // Bottom Left Corner
                        ctx.beginPath();
                        ctx.moveTo(bx + len, by + bh);
                        ctx.lineTo(bx, by + bh);
                        ctx.lineTo(bx, by + bh - len);
                        ctx.stroke();
                        
                        // Bottom Right Corner
                        ctx.beginPath();
                        ctx.moveTo(bx + bw - len, by + bh);
                        ctx.lineTo(bx + bw, by + bh);
                        ctx.lineTo(bx + bw, by + bh - len);
                        ctx.stroke();
                        
                        // Target locked HUD text tag
                        ctx.fillStyle = themeHexColors[activeTheme] || '#00F0FF';
                        ctx.font = 'bold 15px "Share Tech Mono", monospace';
                        ctx.shadowBlur = 0;
                        ctx.fillText(`TARGET LOCKED // ${result.emotion.toUpperCase()} (${(result.confidence * 100).toFixed(0)}%)`, bx, by - 12);
                    }
                } else {
                    document.getElementById('emotion').textContent = "ACQUIRING...";
                    document.getElementById('confidence').textContent = "0.0%";
                    bioBpm.textContent = "72 BPM";
                    bioStress.textContent = "12%";
                    bioPulseBPM = 72;
                    bioCurrentStress = 12;
                }
            } catch (err) {
                console.error("Inference fetch error:", err);
            }
        }
        
        function updateBiometrics(emotion, confidence) {
            // High Stress States
            if (emotion === "Anger" || emotion === "Fear") {
                bioPulseBPM = Math.round(110 + (confidence * 30)); // 110-140 BPM
                bioCurrentStress = Math.round(75 + (confidence * 20)); // 75-95%
            } 
            // Moderate Stress States
            else if (emotion === "Sadness" || emotion === "Disgust") {
                bioPulseBPM = Math.round(55 + (confidence * 10)); // 55-65 BPM
                bioCurrentStress = Math.round(40 + (confidence * 15)); // 40-55%
            }
            // Dynamic Surprise States
            else if (emotion === "Surprise") {
                bioPulseBPM = Math.round(95 + (confidence * 15)); // 95-110 BPM
                bioCurrentStress = Math.round(50 + (confidence * 10)); // 50-60%
            }
            // Steady States
            else {
                bioPulseBPM = Math.round(68 + (confidence * 8)); // 68-76 BPM
                bioCurrentStress = Math.round(10 + (confidence * 8)); // 10-18%
            }
            
            bioBpm.textContent = `${bioPulseBPM} BPM`;
            bioStress.textContent = `${bioCurrentStress}%`;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return HTMLResponse(content=HTML_CONTENT)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
