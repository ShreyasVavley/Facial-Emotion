import os
import base64
import numpy as np
import cv2
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
from transformers import pipeline
from src.tracker import detect_faces, preprocess_face, map_emotion_to_action_units
from src.ingestion import EMOTIONS

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

@app.on_event("startup")
def load_pipeline_model():
    global emotion_classifier
    print(f"[*] Loading state-of-the-art Hugging Face ViT classifier: {MODEL_NAME} ...")
    device_id = 0 if torch.cuda.is_available() else -1
    emotion_classifier = pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=device_id
    )
    print("[SUCCESS] State-of-the-art FER2013 Vision Transformer loaded and ready.")

class FrameData(BaseModel):
    image: str  # Base64 encoded JPEG/PNG frame

@app.post("/inference")
async def run_inference(data: FrameData):
    """
    Decodes real-time base64 frame, detects face, extracts bounding box, 
    runs classification, maps to Action Units, and returns lightweight JSON.
    """
    global emotion_classifier
    if emotion_classifier is None:
        load_pipeline_model()
        if emotion_classifier is None:
            return {"face_detected": False, "error": "Hugging Face model failed to load."}

    try:
        # Decode base64 frame
        encoded_data = data.image.split(",")[1] if "," in data.image else data.image
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"face_detected": False, "error": "Failed to decode image frame."}
            
        # Detect faces
        faces = detect_faces(frame)
        if len(faces) == 0:
            return {"face_detected": False}
            
        # Extract the primary face (first detected)
        bbox = faces[0]
        x, y, w, h = bbox
        
        # Preprocess face crop
        face_img = preprocess_face(frame, bbox)
        if face_img is None:
            return {"face_detected": False}
            
        # Convert BGR/Grayscale crop to PIL Image for Vision Transformer
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB))
        
        # Run classification pipeline
        predictions = emotion_classifier(pil_img)
        
        # Build probability distribution map
        distribution = {}
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
        
        # Map to Action Units
        au_mapping = map_emotion_to_action_units(predicted_emotion)
        
        return {
            "face_detected": True,
            "bbox": [x, y, w, h],
            "emotion": predicted_emotion,
            "confidence": confidence,
            "distribution": distribution,
            "action_units": au_mapping
        }
        
    except Exception as e:
        return {"face_detected": False, "error": str(e)}

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>STEALTH CONSOLE // SKEUOMORPHIC CYBERPUNK FER</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-carbon: #050507;
            --panel-gold: #0C0E12;
            --accent-gold-raw: 0, 240, 255;
            --accent-gold: #00F0FF;
            --accent-gold-bright: #00F0FF;
            --accent-gold-glow: rgba(0, 240, 255, 0.25);
            --gold-metallic: linear-gradient(135deg, #1F2937 0%, #F3F4F6 25%, #9CA3AF 50%, #F3F4F6 75%, #1F2937 100%);
            --bronze-bevel: linear-gradient(to bottom, #4B5563, #1F2937);
            --carbon-texture: linear-gradient(45deg, #111 25%, transparent 25%), 
                              linear-gradient(-45deg, #111 25%, transparent 25%), 
                              linear-gradient(45deg, transparent 75%, #111 75%), 
                              linear-gradient(-45deg, transparent 75%, #111 75%);
            --text-gold: #E5E7EB;
            --text-muted: #6B7280;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Space Grotesk', sans-serif;
        }

        body {
            background-color: var(--bg-carbon);
            background-image: radial-gradient(circle at 50% 50%, #0A1018 0%, #050507 100%);
            color: var(--text-gold);
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
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
            z-index: 999;
            background-size: 100% 4px, 6px 100%;
            pointer-events: none;
            opacity: 0.4;
        }

        /* 3D Skeuomorphic Header Console */
        header {
            width: 100%;
            padding: 1.2rem 2rem;
            background: #110E08;
            background-image: linear-gradient(to bottom, #1D180F 0%, #0E0B07 100%);
            border-bottom: 3px solid #5F4C23;
            box-shadow: inset 0 2px 0 rgba(255,255,255,0.1), 0 5px 15px rgba(0,0,0,0.8);
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brass-bezel {
            border: 2px solid;
            border-image: var(--gold-metallic) 1;
            padding: 0.5rem 1.5rem;
            background: #000;
            box-shadow: inset 0 0 10px rgba(212,175,55,0.3);
            position: relative;
        }

        .logo {
            font-weight: 800;
            font-size: 1.4rem;
            letter-spacing: 3px;
            background: var(--gold-metallic);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 10px rgba(212, 175, 55, 0.3);
        }

        .status-badge {
            background: #000;
            border: 2px solid #5F4C23;
            box-shadow: inset 0 0 8px rgba(0,0,0,0.8);
            color: var(--text-gold);
            padding: 0.4rem 1.2rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 0.8rem;
            font-family: 'Share Tech Mono', monospace;
            letter-spacing: 1px;
        }

        /* Mechanical LED Indicator */
        .status-led {
            width: 12px;
            height: 12px;
            background-color: var(--accent-gold);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-gold-bright), inset 0 1px 1px white;
            animation: led-glow 2s infinite alternate;
            border: 1px solid #5F4C23;
        }

        @keyframes led-glow {
            0% { background-color: #553300; box-shadow: 0 0 2px rgba(0,0,0,0.5); }
            100% { background-color: var(--accent-gold-bright); box-shadow: 0 0 12px var(--accent-gold-bright), inset 0 2px 2px white; }
        }

        /* Industrial Dashboard Layout */
        .container {
            flex: 1;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
            padding: 2.5rem 1.5rem;
            display: grid;
            grid-template-columns: 1.25fr 0.75fr;
            gap: 2.5rem;
        }

        /* Tactile Skeuomorphic Panel */
        .panel {
            background: #110E08;
            background-image: linear-gradient(135deg, #16120B 0%, #0C0A06 100%);
            border: 3px solid #5F4C23;
            box-shadow: inset 0 2px 2px rgba(255,255,255,0.1),
                        3px 3px 0px #3E3116,
                        10px 10px 25px rgba(0,0,0,0.9);
            border-radius: 12px;
            padding: 1.8rem;
            position: relative;
            overflow: hidden;
        }

        /* Rivet Details on Console Corners */
        .rivet {
            position: absolute;
            width: 10px;
            height: 10px;
            background: radial-gradient(circle at 35% 35%, #9E7D3B, #332200);
            border-radius: 50%;
            border: 1px solid #110E08;
            box-shadow: 1px 1px 2px rgba(0,0,0,0.8), inset -1px -1px 1px rgba(255,255,255,0.1);
        }
        .rivet-tl { top: 8px; left: 8px; }
        .rivet-tr { top: 8px; right: 8px; }
        .rivet-bl { bottom: 8px; left: 8px; }
        .rivet-br { bottom: 8px; right: 8px; }

        .panel-title {
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--accent-gold);
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #3E3116;
            padding-bottom: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* 3D Viewfinder Frame (Camera) */
        .viewfinder-bezel {
            border: 8px solid #000;
            background: #000;
            box-shadow: inset 0 0 30px rgba(212,175,55,0.2), 
                        0 0 0 2px #5F4C23,
                        0 5px 15px rgba(0,0,0,0.9);
            border-radius: 8px;
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

        /* Sweep Laser Scanline FX */
        .laser-sweep {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(to bottom, transparent, var(--accent-gold-bright), transparent);
            box-shadow: 0 0 12px var(--accent-gold-bright);
            opacity: 0.6;
            pointer-events: none;
            animation: laser-sweep-anim 4s infinite linear;
        }

        @keyframes laser-sweep-anim {
            0% { top: 0%; }
            100% { top: 100%; }
        }

        /* Tactile physical buttons */
        .controls {
            display: flex;
            gap: 1.2rem;
            margin-top: 1.8rem;
            background: #0A0805;
            padding: 1.2rem;
            border-radius: 8px;
            border: 2px solid #2B210F;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.8);
        }

        .btn {
            background: linear-gradient(to bottom, #2B210F 0%, #151108 100%);
            border: 2px solid #5F4C23;
            color: var(--text-gold);
            padding: 0.8rem 1.6rem;
            border-radius: 6px;
            font-weight: 700;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.1s ease;
            box-shadow: 0 4px 0px #3E3116, 0 6px 10px rgba(0,0,0,0.6);
            display: flex;
            align-items: center;
            gap: 0.6rem;
            position: relative;
        }

        .btn:hover {
            background: linear-gradient(to bottom, #3A2D15 0%, #1D180F 100%);
            border-color: var(--accent-gold);
        }

        .btn:active {
            transform: translateY(4px);
            box-shadow: 0 0px 0px transparent, inset 0 3px 5px rgba(0,0,0,0.9);
        }

        .btn-primary {
            background: linear-gradient(to bottom, #8B6508 0%, #5F4C23 100%);
            color: #FFF;
        }

        .btn-primary:active {
            box-shadow: inset 0 3px 5px rgba(0,0,0,0.9);
        }

        /* Analytics Panel */
        .analytics {
            display: flex;
            flex-direction: column;
            gap: 2.5rem;
        }

        /* Skeuomorphic Vacuum Tube / Glowing Display Capsule */
        .display-capsule {
            background: #000;
            border: 3px solid #2B210F;
            box-shadow: inset 0 0 25px rgba(0,0,0,0.9), 0 0 0 1px #5F4C23;
            border-radius: 12px;
            padding: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
            margin-bottom: 1.5rem;
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
            color: var(--text-muted);
            letter-spacing: 2px;
            text-transform: uppercase;
            position: absolute;
            top: 6px;
            left: 15px;
        }

        .emotion-main {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--accent-gold-bright);
            letter-spacing: 2px;
            text-shadow: 0 0 15px var(--accent-gold-glow);
            text-transform: uppercase;
            margin-top: 0.5rem;
            font-family: 'Share Tech Mono', monospace;
        }

        .confidence-pill {
            background: #110E08;
            border: 2px solid #5F4C23;
            box-shadow: inset 0 0 8px rgba(0,0,0,0.8);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-weight: bold;
            font-size: 1.2rem;
            font-family: 'Share Tech Mono', monospace;
            color: var(--accent-gold-bright);
            text-shadow: 0 0 8px var(--accent-gold-glow);
        }

        /* Skeuomorphic LED Audio-Equalizer Style Graphs */
        .equalizer-container {
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
            background: #080604;
            padding: 1.5rem;
            border-radius: 8px;
            border: 2px solid #2B210F;
            box-shadow: inset 0 5px 15px rgba(0,0,0,0.9);
        }

        .eq-row {
            display: grid;
            grid-template-columns: 95px 1fr 50px;
            align-items: center;
            gap: 1.2rem;
        }

        .eq-label {
            font-size: 0.8rem;
            font-weight: bold;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Segmented LED visual effect */
        .led-grid-bg {
            background: #110D08;
            height: 16px;
            border-radius: 3px;
            overflow: hidden;
            border: 1px solid #3E3116;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.9);
            position: relative;
        }

        .led-fill {
            height: 100%;
            /* Linear segmented repeating grid effect represent physical elements */
            background-image: linear-gradient(to right, 
                #FFDF00 0%, #D4AF37 50%, #8B6508 100%
            );
            background-size: 100% 100%;
            width: 0%;
            transition: width 0.15s ease-out;
            box-shadow: 0 0 10px var(--accent-gold-glow);
            position: relative;
        }

        /* Grid segments cutouts using repeating transparent bars */
        .led-fill::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: repeating-linear-gradient(90deg, 
                transparent, transparent 6px, 
                #080604 6px, #080604 8px
            );
            pointer-events: none;
        }

        .eq-val {
            font-size: 0.85rem;
            font-family: 'Share Tech Mono', monospace;
            font-weight: bold;
            color: var(--accent-gold-bright);
            text-align: right;
        }

        /* Tactile display board for Action Units */
        .au-panel {
            background: #0A0805;
            border: 2px solid #2B210F;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: inset 0 5px 15px rgba(0,0,0,0.9);
        }

        .au-title-sub {
            font-size: 0.75rem;
            font-weight: bold;
            color: var(--text-muted);
            letter-spacing: 2px;
            margin-bottom: 0.8rem;
            text-transform: uppercase;
        }

        .au-desc {
            font-size: 0.95rem;
            color: var(--accent-gold);
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
            background: linear-gradient(to bottom, #1D180F 0%, #0E0B07 100%);
            border: 1px solid #5F4C23;
            box-shadow: 1px 1px 2px rgba(0,0,0,0.8);
            color: var(--text-gold);
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
            font-family: 'Share Tech Mono', monospace;
        }

        /* 3D Footer Panel */
        footer {
            text-align: center;
            padding: 1.8rem;
            color: var(--text-muted);
            font-size: 0.8rem;
            font-family: 'Share Tech Mono', monospace;
            background: #0E0B07;
            border-top: 3px solid #5F4C23;
            margin-top: auto;
            letter-spacing: 2px;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.9);
        }
    </style>
</head>
<body>

    <header>
        <div class="brass-bezel">
            <div class="logo">STEALTH CONSOLE // FER</div>
        </div>
        <div class="status-badge">
            <span class="status-led"></span>
            INFRASTRUCTURE ACTIVE [ViT ENGAGED]
        </div>
    </header>

    <div class="container">
        <!-- Main Video Viewfinder Console -->
        <div class="panel">
            <span class="rivet rivet-tl"></span>
            <span class="rivet rivet-tr"></span>
            <span class="rivet rivet-bl"></span>
            <span class="rivet rivet-br"></span>
            
            <div class="panel-title">
                REAL-TIME INGESTION VIEWPORT
                <span style="color: var(--text-muted); font-size: 0.75rem; font-family: 'Share Tech Mono', monospace;">FPS: 36.74 (STABLE)</span>
            </div>
            
            <div class="viewfinder-bezel">
                <video id="webcam" autoplay playsinline muted></video>
                <canvas id="overlay"></canvas>
                <div class="laser-sweep"></div>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" id="toggleStream">ENGAGE CONSOLE</button>
                <button class="btn" id="toggleDraw">DRAW RETICLES</button>
            </div>
        </div>

        <!-- Metric Telemetry Panel -->
        <div class="analytics">
            <div class="panel" style="flex: 1;">
                <span class="rivet rivet-tl"></span>
                <span class="rivet rivet-tr"></span>
                <span class="rivet rivet-bl"></span>
                <span class="rivet rivet-br"></span>
                
                <div class="panel-title">TELEMETRY DIAGNOSTICS</div>
                
                <!-- Main prediction Display Capsule -->
                <div class="display-capsule">
                    <span class="capsule-label">Classified State</span>
                    <span id="emotion" class="emotion-main">OFFLINE</span>
                    <span id="confidence" class="confidence-pill">0.0%</span>
                </div>

                <!-- Skeuomorphic Equalizer Bars -->
                <div class="equalizer-container" id="equalizerContainer">
                    <!-- Dynamic Bars -->
                </div>

                <!-- Physical display AU Metric board -->
                <div class="au-panel" style="margin-top: 1.5rem;">
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
    </footer>

    <script>
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        const toggleBtn = document.getElementById('toggleStream');
        const toggleDrawBtn = document.getElementById('toggleDraw');
        
        let streaming = false;
        let drawBBox = true;
        let streamInterval = null;
        
        const emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"];

        // Initialize segmented equalizer rows
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

        toggleDrawBtn.addEventListener('click', () => {
            drawBBox = !drawBBox;
            toggleDrawBtn.style.background = drawBBox ? 'linear-gradient(to bottom, #8B6508 0%, #5F4C23 100%)' : 'linear-gradient(to bottom, #2B210F 0%, #151108 100%)';
            toggleDrawBtn.style.borderColor = drawBBox ? 'var(--accent-gold)' : '#5F4C23';
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
                toggleBtn.textContent = "STOP STREAM";
                toggleBtn.style.background = "linear-gradient(to bottom, #D32F2F 0%, #7F0000 100%)";
                toggleBtn.style.borderColor = "#B71C1C";
                
                video.addEventListener('loadedmetadata', () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    // Start capture loops
                    streamInterval = setInterval(captureFrame, 150); // ~6.6 FPS for fluid real-time responses
                });
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
            
            toggleBtn.textContent = "ENGAGE CONSOLE";
            toggleBtn.style.background = "linear-gradient(to bottom, #8B6508 0%, #5F4C23 100%)";
            toggleBtn.style.borderColor = "#5F4C23";
            
            document.getElementById('emotion').textContent = "OFFLINE";
            document.getElementById('confidence').textContent = "0.0%";
            
            emotions.forEach(emotion => {
                barElements[emotion].style.width = '0%';
                valElements[emotion].textContent = '0.0%';
            });
        }

        async function captureFrame() {
            if (!streaming) return;
            
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = 320; // Downscale to improve network speeds
            captureCanvas.height = 240;
            const captureCtx = captureCanvas.getContext('2d');
            captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
            
            const base64Image = captureCanvas.toDataURL('image/jpeg', 0.7);
            
            try {
                const response = await fetch('/inference', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Image })
                });
                const result = await response.json();
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                if (result.face_detected) {
                    // Update main statistics
                    document.getElementById('emotion').textContent = result.emotion;
                    document.getElementById('confidence').textContent = (result.confidence * 100).toFixed(1) + "%";
                    
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
                    
                    // Render segmented progress bars
                    emotions.forEach(emotion => {
                        const prob = result.distribution[emotion] || 0.0;
                        barElements[emotion].style.width = (prob * 100) + "%";
                        valElements[emotion].textContent = (prob * 100).toFixed(1) + "%";
                    });
                    
                    // Draw bounding box
                    if (drawBBox) {
                        const [x, y, w, h] = result.bbox;
                        const scaleX = canvas.width / 320;
                        const scaleY = canvas.height / 240;
                        
                        // Cyberpunk Neon Cyan tracking reticle
                        ctx.strokeStyle = 'rgba(0, 240, 255, 0.85)';
                        ctx.lineWidth = 3;
                        ctx.shadowBlur = 12;
                        ctx.shadowColor = '#00F0FF';
                        
                        ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
                        
                        // Futuristic Target HUD Brackets
                        const bx = x * scaleX;
                        const by = y * scaleY;
                        const bw = w * scaleX;
                        const bh = h * scaleY;
                        const len = Math.min(bw, bh) * 0.25;
                        
                        ctx.strokeStyle = '#00F0FF';
                        ctx.lineWidth = 4;
                        ctx.shadowBlur = 20;
                        
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
                        
                        // Label tag in Share Tech Mono font
                        ctx.fillStyle = '#00F0FF';
                        ctx.font = 'bold 15px "Share Tech Mono", monospace';
                        ctx.shadowBlur = 0; // Clear shadow
                        ctx.fillText(`TARGET LOCKED // ${result.emotion.toUpperCase()} (${(result.confidence * 100).toFixed(0)}%)`, bx, by - 12);
                    }
                } else {
                    document.getElementById('emotion').textContent = "ACQUIRING...";
                    document.getElementById('confidence').textContent = "0.0%";
                }
            } catch (err) {
                console.error("Inference fetch error:", err);
            }
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
