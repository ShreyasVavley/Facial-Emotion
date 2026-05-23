import os
import base64
import json
import asyncio
import numpy as np
import cv2

# Import app, schemas, and synthetic generators
from src.app import run_inference, FrameData, load_pipeline_model
from src.ingestion import draw_synthetic_emotion

async def run_integration_test():
    print("====================================================")
    print("PHASE 4: DIRECT E2E PIPELINE INTEGRATION TEST")
    print("====================================================")
    
    # 1. Generate synthetic frame representation
    img = draw_synthetic_emotion("Happiness")
    # Convert grayscale to color since webcam feeds are in BGR/RGB color
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    base64_img = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    
    print("[*] Generated synthetic base64 image representing 'Happiness'")
    
    # 2. Boot up pipeline and load weights
    load_pipeline_model()
    
    # 3. Create input model frame payload
    payload = FrameData(image=base64_img)
    
    # 4. Invoke the FastAPI inference endpoint directly
    print("[*] Invoking uvicorn-ready '/inference' pipeline endpoint...")
    
    # Run async function
    result = await run_inference(payload)
    
    # 5. Output minimal production JSON log as requested
    print("\n----------------------------------------------------")
    print("MINIMAL PRODUCTION JSON INFERENCE LOG:")
    print("----------------------------------------------------")
    print(json.dumps(result, indent=2))
    print("----------------------------------------------------")
    
    # 6. Assertions for Target State Verification
    assert result.get("face_detected") is True, "Error: Face localization failed!"
    assert result.get("emotion") in ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"], f"Error: Invalid emotion: {result.get('emotion')}"
    assert result.get("confidence") > 0.0, "Error: Invalid confidence score"
    assert "distribution" in result, "Error: Distribution map missing in JSON output."
    assert "action_units" in result, "Error: FACS Action Units missing."
    assert "bbox" in result, "Error: Bounding box coordinates missing."
    
    print("\n[OK] End-to-end integration and JSON logs successfully verified.")
    print("====================================================")
    print("PHASE 4 STATE VALIDATION: SUCCESSFUL")
    print("====================================================")

if __name__ == "__main__":
    asyncio.run(run_integration_test())
