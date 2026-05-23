import os
import time
import torch
import numpy as np
import cv2
from src.models import EmotionCNN
from src.tracker import preprocess_face, detect_faces
from src.ingestion import draw_synthetic_emotion

def benchmark_pipeline():
    print("====================================================")
    print("NEURAL NEXUS // FER PIPELINE EFFICIENCY BENCHMARK")
    print("====================================================")
    
    # 1. Measure Startup & Weight Load Latency
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=7).to(device)
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    load_latency = (time.time() - start_time) * 1000
    print(f"[*] Pipeline Engine Load Latency: {load_latency:.2f} ms")
    
    # 2. Benchmarking Preprocessing
    print("[*] Generating synthetic test frame stream...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw face in center
    cv2.circle(frame, (320, 240), 100, (255, 255, 255), -1)
    
    # Measure face crop extraction latency
    start_time = time.time()
    for _ in range(50):
        faces = detect_faces(frame)
        if len(faces) > 0:
            preprocess_face(frame, faces[0])
    preproc_latency = ((time.time() - start_time) / 50) * 1000
    print(f"[*] Average Face Preprocessing & Tracking Latency: {preproc_latency:.3f} ms")
    
    # 3. Benchmarking Pure Model Inference
    # Create normalized input tensor (B x C x H x W)
    dummy_input = torch.randn(1, 1, 48, 48).to(device)
    
    print("[*] Running 100 warm-up runs to stabilize graph caches...")
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
            
    print("[*] Initiating 500-pass forward inference benchmark...")
    inference_times = []
    
    with torch.no_grad():
        for _ in range(500):
            t0 = time.time()
            _ = model(dummy_input)
            inference_times.append((time.time() - t0) * 1000)
            
    avg_inf = np.mean(inference_times)
    min_inf = np.min(inference_times)
    max_inf = np.max(inference_times)
    std_inf = np.std(inference_times)
    fps = 1000.0 / avg_inf
    
    print("\n----------------------------------------------------")
    print("INFERENCE SPEED STATISTICS (BATCH SIZE = 1):")
    print("----------------------------------------------------")
    print(f"Device:                 {device.type.upper()}")
    print(f"Average Latency:        {avg_inf:.3f} ms")
    print(f"Minimum Latency:        {min_inf:.3f} ms")
    print(f"Maximum Latency:        {max_inf:.3f} ms")
    print(f"Standard Deviation:     {std_inf:.3f} ms")
    print(f"Throughput Capacity:    {fps:.2f} FPS")
    print("----------------------------------------------------")
    
    # 4. Total Pipeline End-to-End Latency
    e2e_avg = preproc_latency + avg_inf
    e2e_fps = 1000.0 / e2e_avg
    print(f"\n[*] Total E2E Local Loop Latency: {e2e_avg:.3f} ms")
    print(f"[*] Max Theoretical Live Capture Rate: {e2e_fps:.2f} FPS")
    print("====================================================")
    print("EFFICIENCY EVALUATION REPORT: COMPLETED SUCCESSFULLY")
    print("====================================================")

if __name__ == "__main__":
    benchmark_pipeline()
