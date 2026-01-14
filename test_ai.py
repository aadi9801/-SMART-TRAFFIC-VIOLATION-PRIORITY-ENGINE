from ultralytics import YOLO
import cv2
import numpy as np

print("Testing imports...")
try:
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")
    
    # Create a dummy image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(img, verbose=False)
    print("Inference successful!")
except Exception as e:
    print(f"Error: {e}")
