# STEALTH CONSOLE // FACIAL EMOTION RECOGNITION (FER) PIPELINE

A state-of-the-art, high-precision **Facial Emotion Recognition (FER)** system combining a pre-trained **Hugging Face Vision Transformer (ViT)** classifier with an interactive, highly tactile **Skeuomorphic Cyberpunk Console** served locally via **FastAPI** and deployed globally on **Hugging Face Spaces**!

---

## 📸 Cyberpunk Console Overview

The frontend dashboard is designed using a high-fidelity **Skeuomorphic Cyberpunk** theme. It integrates physical skeuomorphic hardware elements (brushed steel panels, linear bevel profiles, brass rivets, 3D rocker switches, rotating knobs, vacuum tubes, and vector screens) with cyberpunk HUD telemetry diagnostics.

### Core Interactive Features:
1.  **🎛️ Dual-Model Inference Switch:** A physical 3D rocker switch that swaps the backend pipeline between the heavy **Vision Transformer (ViT)** (~343MB) and the ultra-lightweight **Custom local CNN** (10.6MB) on-the-fly, displaying live benchmark latency side-by-side.
2.  **🔊 Web Audio Sonification Deck:** Immersive sound deck featuring deep 45Hz reactor hum oscillators, sweeping radar audio clicks, dynamic target-pitch chimes, and a **robot vocal announcer** utilizing native SpeechSynthesis to warn when high-stress targets are locked.
3.  **💓 Biometric ECG Telemetry:** An active vector EKG viewport displaying a canvas-drawn phosphor grid. The ECG sweep speed, amplitude, and QRS pulse rate scale dynamically with your emotions (spiking up to 140 BPM for anger/fear, stabilizing for neutral).
4.  **🔬 Calibration board & CRT Viewfinder:** A slide-out panel with metallic range sliders to calibrate face tracking thresholds (`scaleFactor` and `minNeighbors`) in real-time, accompanied by a circular, curved CRT viewfinder displaying the grayscaled preprocessing face crop fed into the model.
5.  **🎨 Multi-Theme Selector Dial:** A physical selection knob that rotates to instantly reskin the entire interface between 4 neon themes: `Stealth Cyan`, `Deus Ex Amber`, `Matrix Green`, and `Crimson Alert` via smooth CSS variables updates.
6.  **💽 Memory Cartridge Floppy Bay:** A skeuomorphic floppy slot. Mounting a magnetic cartridge logs all classified frames, latencies, and biometrics to be ejected and downloaded as a formatted JSON telemetry session file.

---

## 🌐 Live Cloud Deployment

The production build of this application is fully Dockerized and deployed live on Hugging Face Spaces:

🚀 **[Access Live Hugging Face Space Console](https://huggingface.co/spaces/ShreyasVavley/facial-emotion-console)**

*(Note: Ensure you allow webcam permissions and click "SONIC DECK" to engage audio feedback).*

---

## 📁 Repository Structure

```
face/
├── .github/
│   └── workflows/
│       └── hf_sync.yml         # CI/CD GitHub Actions workflow (Auto-syncs to Hugging Face)
├── .venv/                      # Isolated virtual environment
├── src/
│   ├── ingestion.py            # Image generator and stratified split logic
│   ├── models.py               # Custom CNN PyTorch architecture
│   ├── tracker.py              # Face detection, fallback, and Action Units mapper
│   └── app.py                  # FastAPI server & Skeuomorphic Console (HTML/CSS)
├── models/
│   └── best_model.pth          # Custom CNN trained weights (10.6MB, tracked by Git LFS)
├── requirements.txt            # Package dependencies
├── Dockerfile                  # Multi-stage production Docker container configuration
├── verify_phase1.py            # Phase 1 shape and split validation
├── verify_phase2.py            # Phase 2 graph compilation validation
├── verify_phase4.py            # Phase 4 end-to-end integration validation
├── verify_efficiency.py        # 500-pass efficiency benchmark script
└── README.md                   # System documentation
```

---

## 🚀 Quick Start Guide

### 1. Provision the Environment
Configure a local virtual environment and install dependencies:
```powershell
# Create venv
python -m venv .venv

# Install dependencies
.venv\Scripts\python -m pip install -r requirements.txt
```

### 2. Launch the Console Locally
Boot up the FastAPI server:
```powershell
.venv\Scripts\python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your web browser and engage the webcam.

---

## ⚙️ CI/CD Auto-Sync Deployment
Our GitHub Actions workflow automatically syncs and deploys the codebase to Hugging Face Spaces on every push:
1. Create a Space on Hugging Face using the **Docker** SDK and naming it `facial-emotion-console`.
2. Generate a **Write Token** under your Hugging Face Settings.
3. Add the token to your GitHub repository secrets as `HF_TOKEN`.

---

## 🔍 Validation Tests

Run local validation scripts to inspect mathematical and graph correctness:
```powershell
# Phase 1 Validation (Sets, leakage, grayscale bounds)
.venv\Scripts\python verify_phase1.py

# Phase 2 Validation (Graph compile, softmax probability, shapes)
.venv\Scripts\python verify_phase2.py

# Phase 4 Direct Integration Test (Asserts full API schema & crop outputs)
.venv\Scripts\python verify_phase4.py

# Run CPU Efficiency Benchmark (Latency, FPS, standard deviation)
.venv\Scripts\python verify_efficiency.py
```
