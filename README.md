# STEALTH CONSOLE // FACIAL EMOTION RECOGNITION (FER) PIPELINE

A state-of-the-art, high-precision **Facial Emotion Recognition (FER)** system combining a pre-trained **Hugging Face Vision Transformer (ViT)** classifier with an interactive, highly tactile **Skeuomorphic Cyberpunk Console** served locally via **FastAPI**!

---

## 📸 Console Interface Overview

The frontend dashboard is designed using a **Stealth Steel & Cyberpunk Cyan** theme. It integrates physical skeuomorphic elements (tactile 3D metallic buttons, carbon panels, brass screws, and segmented LEDs) with cyberpunk HUD telemetry diagnostics.

### Key Visual & CSS Effects:
*   **Tactile 3D Controls:** Physical switches and buttons that physically depress when clicked (`transform: translateY(4px)`) and transition between realistic inset highlights and metallic bevel shadows.
*   **Brushed Steel Bezels:** The main viewport and header are framed inside high-fidelity 3D metallic profiles constructed with custom linear gradients.
*   **Neon Laser Scanlines:** An ambient cyan-colored vertical scanning laser sweeps the camera viewport continuously to indicate active face tracking.
*   **HUD Reticle Corners:** Cyberpunk-style corner brackets frame detected faces on the viewport canvas, accompanied by real-time `TARGET LOCKED` telemetry tags.
*   **Segmented LED Bars:** Horizontally repeating LED cutouts that light up progressively inside custom slots to represent classification probability maps.

---

## ⚙️ Core Technical Features

1.  **State-of-the-Art Classification:** Leverages a pre-trained Hugging Face Vision Transformer (`dima806/facial_emotions_image_detection`) fine-tuned on the massive **FER2013** dataset (35,000+ real human faces), providing high precision on actual webcam feeds.
2.  **Facial Action Coding System (FACS) Binding:** Classifications are dynamically mapped to active **Action Units (AUs)** (e.g. brow tension, lip corners pullers), translating raw logits into descriptive human motor controls.
3.  **Face Localization Fallback:** Employs an OpenCV Haar Cascade frontal face detector inside the viewport. If poor lighting or angling prevents face detection, the tracker automatically deploys a central 85% region crop fallback to maintain fluid, non-blocking telemetry.
4.  **Mathematical Leakage Isolation:** Set-theoretic search scripts mathematically prove exactly 0 overlapping pixel samples between training splits to ensure zero leakage.

---

## 📈 Performance & Efficiency Metrics

The local E2E loop latency easily completes under **30 ms**, exceeding standard webcam feed frames (33 ms at 30 FPS) for smooth real-time response:

| Benchmark Stage | Latency Metrics | Target Performance |
| :--- | :--- | :--- |
| **Pipeline Engine Load** | `44.35 ms` | Weights & Graph Startup Boot Time |
| **Face Preprocessing & Tracking** | `20.346 ms` | Haar Cascade Detection, Scaling & Fallback |
| **Pure CNN Model Inference** | `6.874 ms` | Single Frame Feedforward Pass Time |
| **End-to-End Local Loop** | `27.220 ms` | Complete Capture-to-Log Pipeline Delay |

*   **Maximum Standalone Inference Capacity:** **`145.47 FPS`** (CPU)
*   **Maximum Theoretical Live Capture Rate:** **`36.74 FPS`** (E2E Loop)

---

## 📁 Repository Structure

```
face/
├── .venv/                      # Isolated virtual environment
├── src/
│   ├── ingestion.py            # Image generator and stratified split logic
│   ├── models.py               # Custom CNN PyTorch architecture
│   ├── tracker.py              # Face detection, fallback, and Action Units mapper
│   └── app.py                  # FastAPI server & Skeuomorphic Console (HTML/CSS)
├── models/
│   └── best_model.pth          # Custom CNN trained weights (100% convergence)
├── requirements.txt            # Package dependencies
├── verify_phase1.py            # Phase 1 shape and split validation
├── verify_phase2.py            # Phase 2 graph compilation validation
├── verify_phase4.py            # Phase 4 end-to-end integration validation
├── verify_efficiency.py        # 500-pass efficiency benchmark script
└── README.md                   # System documentation
```

---

## 🚀 Quick Start Guide

### 1. Provision the Environment
Configure a local virtual environment and install standard scientific, computer vision, and FastAPI packages:
```powershell
# Create venv
python -m venv .venv

# Install dependencies
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install transformers huggingface-hub
```

### 2. Launch the Console
Boot up the stateful FastAPI deployment server:
```powershell
.venv\Scripts\python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

### 3. Open the Console Dashboard
Open your web browser and navigate to:
**[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

*   Click **ENGAGE CONSOLE** to start your camera stream.
*   Interact with **DRAW RETICLES** to toggle HUD bounding boxes on/off.

---

## 🔍 Validation Tests

Run local validation scripts using your python virtual environment to inspect mathematical and graph correctness:
```powershell
# Phase 1 Validation (Sets, leakage, grayscale bounds)
.venv\Scripts\python verify_phase1.py

# Phase 2 Validation (Graph compile, softmax probability, shapes)
.venv\Scripts\python verify_phase2.py

# Phase 4 Direct Integration Test (REST schema assertions)
.venv\Scripts\python verify_phase4.py

# Run CPU Efficiency Benchmark (Latency, FPS, standard deviation)
.venv\Scripts\python verify_efficiency.py
```
