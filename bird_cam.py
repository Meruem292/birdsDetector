import os
import time
import threading
import urllib.request
import urllib.error
import cv2
from ultralytics import YOLO
import numpy as np

# choose smallest available model in repo
if os.path.exists("yolov8n.pt"):
    model_path = "yolov8n.pt"
elif os.path.exists("yolov8m.pt"):
    model_path = "yolov8m.pt"
else:
    model_path = "yolov8l.pt"

print(f"Loading model: {model_path}")
model = YOLO(model_path, task="detect")
model.to('cpu') 

camera_env = os.getenv('CAMERA', '0')
try:
    camera_arg = int(camera_env)
except Exception:
    camera_arg = camera_env

cap = cv2.VideoCapture(camera_arg)
print(f"Using camera: {camera_env}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

api_url = "https://vercel.app"
last_api_call = 0.0
# SET COOLDOWN TO 60 SECONDS (1 MINUTE)
cooldown_seconds = 60.0

process_every_n_frames = 3  
imgsz = 320
conf_thres = 0.25
iou_thres = 0.45
target_class = 14  # bird class

lock = threading.Lock()

def call_api_async(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            response.read()
        print(f"\n[!!!] BIRD DETECTED! API called: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"\n[!] API call failed: {e}")

print("--- GO SIGNAL: SCANNING STARTED ---")
print(f"Cooldown set to {cooldown_seconds} seconds.")

frame_idx = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        frame_idx += 1

        if frame_idx % process_every_n_frames == 0:
            # Heartbeat print to show active scanning
            if frame_idx % 30 == 0:
                print(".", end="", flush=True) 

            results = model(frame, imgsz=imgsz, conf=conf_thres, iou=iou_thres, classes=[target_class], verbose=False)

            # Check if a bird was found in this frame
            if len(results) > 0 and len(results[0].boxes) > 0:
                now = time.time()
                with lock:
                    # Only call API if 1 minute has passed since the last call
                    if now - last_api_call >= cooldown_seconds:
                        threading.Thread(target=call_api_async, args=(api_url,), daemon=True).start()
                        last_api_call = now
                    else:
                        remaining = int(cooldown_seconds - (now - last_api_call))
                        # Optional: prints remaining cooldown if you are watching the logs
                        print(f" (Bird seen, but on cooldown: {remaining}s)", end="", flush=True)

        # GUI is disabled for Headless OpenCV stability
finally:
    print("\nStopping...")
    cap.release()