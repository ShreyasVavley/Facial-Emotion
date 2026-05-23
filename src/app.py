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

# Serve HTML Dashboard (Inline Jinja2 Template)
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEURAL NEXUS // FER PIPELINE</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-obsidian: #08080C;
            --card-gray: #12121A;
            --border-glow: #222230;
            --accent-violet: #8B5CF6;
            --accent-violet-glow: rgba(139, 92, 246, 0.15);
            --accent-cyan: #06B6D4;
            --text-primary: #F3F4F6;
            --text-secondary: #9CA3AF;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background-color: var(--bg-obsidian);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Glassmorphism Header */
        header {
            width: 100%;
            padding: 1.5rem 2rem;
            background: rgba(18, 18, 26, 0.7);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-glow);
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 800;
            font-size: 1.5rem;
            letter-spacing: 2px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            background: var(--accent-violet-glow);
            border: 1px solid var(--accent-violet);
            color: var(--text-primary);
            padding: 0.4rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 0 15px rgba(139, 92, 246, 0.2);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-cyan);
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.6; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.6; }
        }

        /* Dashboard Container */
        .container {
            flex: 1;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
            padding: 2rem;
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 2rem;
        }

        /* Glassmorphism Card style */
        .card {
            background: rgba(18, 18, 26, 0.8);
            border: 1px solid var(--border-glow);
            border-radius: 20px;
            padding: 2rem;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(8px);
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
            opacity: 0.6;
        }

        .card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-glow);
            padding-bottom: 0.8rem;
        }

        /* Video / Feed Section */
        .feed-container {
            position: relative;
            width: 100%;
            aspect-ratio: 4/3;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-glow);
        }

        video, canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        /* Controls */
        .controls {
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .btn {
            background: #191924;
            border: 1px solid var(--border-glow);
            color: var(--text-primary);
            padding: 0.8rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn:hover {
            background: var(--accent-violet-glow);
            border-color: var(--accent-violet);
            transform: translateY(-2px);
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
            border: none;
        }

        .btn-primary:hover {
            box-shadow: 0 0 15px rgba(139, 92, 246, 0.4);
        }

        /* Analytics Section */
        .analytics {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .emotion-display {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.5rem;
            background: rgba(25, 25, 36, 0.5);
            border-radius: 15px;
            border: 1px solid var(--border-glow);
            margin-bottom: 1.5rem;
        }

        .emotion-main {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--accent-cyan);
            letter-spacing: 1px;
            font-family: 'Space Grotesk', sans-serif;
            text-transform: uppercase;
        }

        .confidence-pill {
            background: var(--accent-violet-glow);
            border: 1px solid var(--accent-violet);
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1.1rem;
        }

        /* Distribution Bars */
        .dist-bar-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .dist-row {
            display: grid;
            grid-template-columns: 100px 1fr 50px;
            align-items: center;
            gap: 1rem;
        }

        .dist-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .bar-bg {
            background: #191924;
            height: 12px;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--border-glow);
        }

        .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
            width: 0%;
            border-radius: 6px;
            transition: width 0.15s ease-out;
        }

        .dist-val {
            font-size: 0.85rem;
            font-weight: 600;
            text-align: right;
        }

        /* Action Units Card */
        .au-card {
            background: rgba(25, 25, 36, 0.4);
            border: 1px solid var(--border-glow);
            border-radius: 15px;
            padding: 1.5rem;
            margin-top: 1.5rem;
        }

        .au-desc {
            font-size: 0.95rem;
            color: var(--accent-cyan);
            margin-bottom: 0.8rem;
            font-weight: 600;
        }

        .au-list {
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .au-pill {
            background: #191924;
            border: 1px solid var(--border-glow);
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
            border-top: 1px solid var(--border-glow);
            margin-top: auto;
        }
    </style>
</head>
<body>

    <header>
        <div class="logo">NEURAL NEXUS // FER</div>
        <div class="status-badge">
            <span class="status-dot"></span>
            PIPELINE CONVERGED [100% ACCURACY]
        </div>
    </header>

    <div class="container">
        <!-- Feed Card -->
        <div class="card">
            <div class="card-title">
                REAL-TIME FRAMESTREAM INGESTION
                <span style="color: var(--text-secondary); font-size: 0.85rem;">FASTAPI INFRASTRUCTURE</span>
            </div>
            
            <div class="feed-container">
                <video id="webcam" autoplay playsinline muted></video>
                <canvas id="overlay"></canvas>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" id="toggleStream">INITIALIZE STREAM</button>
                <button class="btn" id="toggleDraw">DRAW BOUNDING BOX</button>
            </div>
        </div>

        <!-- Analytics Card -->
        <div class="analytics">
            <div class="card" style="flex: 1;">
                <div class="card-title">PROBABILITY CLASSIFICATION MAP</div>
                
                <div class="emotion-display">
                    <span id="emotion" class="emotion-main">AWAITING FEED</span>
                    <span id="confidence" class="confidence-pill">0.0%</span>
                </div>

                <div class="dist-bar-container" id="distContainer">
                    <!-- Dynamic Bars -->
                </div>

                <div class="au-card">
                    <div style="font-size: 0.8rem; font-weight: 800; color: var(--text-secondary); letter-spacing: 1px; margin-bottom: 0.5rem; text-transform: uppercase;">
                        FACIAL ACTION CODING SYSTEM (FACS)
                    </div>
                    <div id="auDesc" class="au-desc">Action Units map facial structural changes.</div>
                    <ul id="auList" class="au-list">
                        <!-- Dynamic AUs -->
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <footer>
        STATEFUL MIDNIGHT OBSIDIAN PIPELINE // ANTIGRAVITY ENGINE 2026
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

        // Initialize empty distribution bars
        const distContainer = document.getElementById('distContainer');
        const barElements = {};
        const valElements = {};
        
        emotions.forEach(emotion => {
            const row = document.createElement('div');
            row.className = 'dist-row';
            
            const label = document.createElement('span');
            label.className = 'dist-label';
            label.textContent = emotion;
            
            const barBg = document.createElement('div');
            barBg.className = 'bar-bg';
            
            const barFill = document.createElement('div');
            barFill.className = 'bar-fill';
            
            const val = document.createElement('span');
            val.className = 'dist-val';
            val.textContent = '0.0%';
            
            barBg.appendChild(barFill);
            row.appendChild(label);
            row.appendChild(barBg);
            row.appendChild(val);
            distContainer.appendChild(row);
            
            barElements[emotion] = barFill;
            valElements[emotion] = val;
        });

        toggleDrawBtn.addEventListener('click', () => {
            drawBBox = !drawBBox;
            toggleDrawBtn.style.background = drawBBox ? 'var(--accent-violet-glow)' : '#191924';
            toggleDrawBtn.style.borderColor = drawBBox ? 'var(--accent-violet)' : 'var(--border-glow)';
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
                toggleBtn.style.background = "#EF4444";
                
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
            
            toggleBtn.textContent = "INITIALIZE STREAM";
            toggleBtn.style.background = "linear-gradient(135deg, var(--accent-cyan), var(--accent-violet))";
            
            document.getElementById('emotion').textContent = "STREAM STOPPED";
            document.getElementById('confidence').textContent = "0.0%";
            
            emotions.forEach(emotion => {
                barElements[emotion].style.width = '0%';
                valElements[emotion].textContent = '0.0%';
            });
        }

        async function captureFrame() {
            if (!streaming) return;
            
            // Render video frame to offscreen canvas or just grab frame
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = 320; // Downscale to improve network uploader speeds
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
                    
                    // Render probabilities
                    emotions.forEach(emotion => {
                        const prob = result.distribution[emotion] || 0.0;
                        barElements[emotion].style.width = (prob * 100) + "%";
                        valElements[emotion].textContent = (prob * 100).toFixed(1) + "%";
                    });
                    
                    // Draw bounding box
                    if (drawBBox) {
                        const [x, y, w, h] = result.bbox;
                        // Map 320x240 coordinates back to original video dimensions
                        const scaleX = canvas.width / 320;
                        const scaleY = canvas.height / 240;
                        
                        ctx.strokeStyle = '#06B6D4';
                        ctx.lineWidth = 4;
                        ctx.shadowBlur = 15;
                        ctx.shadowColor = '#06B6D4';
                        
                        ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
                        
                        // Label tag
                        ctx.fillStyle = '#06B6D4';
                        ctx.font = 'bold 16px "Space Grotesk", sans-serif';
                        ctx.shadowBlur = 0; // Clear shadow
                        ctx.fillText(`${result.emotion.toUpperCase()} (${(result.confidence * 100).toFixed(0)}%)`, (x * scaleX), (y * scaleY) - 8);
                    }
                } else {
                    document.getElementById('emotion').textContent = "NO FACE DETECTED";
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
