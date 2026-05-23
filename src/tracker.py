import cv2
import numpy as np

# Load pre-trained Haar Cascade face detector from OpenCV's internal data directory
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    """
    Detects faces in a frame using Haar Cascades.
    If no face is detected, returns a fallback bounding box covering the central 85% of the image.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Return detected faces if found
    if len(faces) > 0:
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        
    # Robust Fallback: Return central 85% area of the frame
    h_f, w_f = frame.shape[:2]
    cw, ch = int(w_f * 0.85), int(h_f * 0.85)
    cx, cy = (w_f - cw) // 2, (h_f - ch) // 2
    return [(cx, cy, cw, ch)]

def preprocess_face(frame, bbox):
    """
    Crops, grayscales, resizes, and equalizes the face region.
    Args:
        frame (np.ndarray): Original BGR/RGB frame
        bbox (tuple): (x, y, w, h) bounding box
    Returns:
        np.ndarray: Equalized 48x48 grayscale face image, or None if invalid crop
    """
    x, y, w, h = bbox
    height, width = frame.shape[:2]
    
    # Boundary validation
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return None
        
    # Crop face
    face_crop = frame[y1:y2, x1:x2]
    
    # Convert to grayscale if BGR
    if len(face_crop.shape) == 3:
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_crop
        
    # Resize to 48x48
    face_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    
    # Histogram Equalization to normalize contrast
    face_equalized = cv2.equalizeHist(face_resized)
    
    return face_equalized

# Facial Action Units (AUs) mapping to structural transformations
# (FACS - Facial Action Coding System rules)
EMOTION_TO_AU = {
    "Happiness": {
        "description": "Duchenne Smile",
        "units": ["AU6 (Cheek Raiser)", "AU12 (Lip Corner Puller)"]
    },
    "Sadness": {
        "description": "Frowning & Inner Brow Elevation",
        "units": ["AU1 (Inner Brow Raiser)", "AU4 (Brow Lowerer)", "AU15 (Lip Corner Depressor)"]
    },
    "Anger": {
        "description": "Glaring & Lip Tightening",
        "units": ["AU4 (Brow Lowerer)", "AU5 (Upper Lid Raiser)", "AU7 (Lid Tightener)", "AU23 (Lip Tightener)"]
    },
    "Surprise": {
        "description": "Gasping & Wide Eyes",
        "units": ["AU1 (Inner Brow Raiser)", "AU2 (Outer Brow Raiser)", "AU5 (Upper Lid Raiser)", "AU26 (Jaw Drop)"]
    },
    "Fear": {
        "description": "Tense Eyes & Lip Stretching",
        "units": ["AU1 (Inner Brow Raiser)", "AU2 (Outer Brow Raiser)", "AU4 (Brow Lowerer)", "AU5 (Upper Lid Raiser)", "AU20 (Lip Stretcher)", "AU26 (Jaw Drop)"]
    },
    "Disgust": {
        "description": "Nose Wrinkling & Upper Lip Raiser",
        "units": ["AU9 (Nose Wrinkler)", "AU15 (Lip Corner Depressor)", "AU17 (Chin Raiser)"]
    },
    "Neutral": {
        "description": "Resting Face",
        "units": ["No active Action Units detected."]
    }
}

def map_emotion_to_action_units(emotion_name):
    """
    Binds the predicted classification feature maps to corresponding Facial Action Units (AUs).
    """
    return EMOTION_TO_AU.get(emotion_name, {
        "description": "Unknown",
        "units": []
    })
