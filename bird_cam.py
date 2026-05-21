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

api_url = "https://agri-sound.vercel.app/api/play"
last_api_call = 0.0
cooldown_seconds = 30.0

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

frame_idx = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        frame_idx += 1

        # Only run detection on a subset of frames
        if frame_idx % process_every_n_frames == 0:
            # Heartbeat print: so you know it's still scanning
            if frame_idx % 30 == 0:
                print(".", end="", flush=True) # Prints a dot every few seconds

            results = model(frame, imgsz=imgsz, conf=conf_thres, iou=iou_thres, classes=[target_class], verbose=False)

            if len(results) > 0 and len(results[0].boxes) > 0:
                now = time.time()
                with lock:
                    if now - last_api_call >= cooldown_seconds:
                        threading.Thread(target=call_api_async, args=(api_url,), daemon=True).start()
                        last_api_call = now

        # GUI DISABLED FOR REMOTE PI (prevents crashes)
        # cv2.imshow("Bird Detection (Pi4)", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
finally:
    print("\nStopping...")
    cap.release()
    # cv2.destroyAllWindows()