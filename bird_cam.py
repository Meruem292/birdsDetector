import os
import time
import threading
import urllib.request
import urllib.error

import cv2
from ultralytics import YOLO
import numpy as np

# Raspberry Pi 4 friendly settings
# - prefer small model if available
# - lower input size
# - skip frames to reduce CPU
# - call API in background thread and show cooldown overlay

# choose smallest available model in repo
if os.path.exists("yolov8n.pt"):
    model_path = "yolov8n.pt"
elif os.path.exists("yolov8m.pt"):
    model_path = "yolov8m.pt"
else:
    model_path = "yolov8l.pt"

model = YOLO(model_path, task="detect")

# Select camera from env `CAMERA` (index like 0,1 or path like /dev/video0)
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

# Performance tuning
process_every_n_frames = 3  # skip frames
imgsz = 320
conf_thres = 0.25
iou_thres = 0.45
target_class = 14  # bird class

lock = threading.Lock()

def call_api_async(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            response.read()
        print(f"API called: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"API call failed: {e}")


frame_idx = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated = frame

        # Only run detection on a subset of frames to reduce CPU usage
        if frame_idx % process_every_n_frames == 0:
            results = model(frame, imgsz=imgsz, conf=conf_thres, iou=iou_thres, classes=[target_class], verbose=False)
            annotated = results[0].plot() if len(results) > 0 else frame

            # If a bird is detected, trigger API (with cooldown)
            if len(results) > 0 and len(results[0].boxes) > 0:
                now = time.time()
                with lock:
                    if now - last_api_call >= cooldown_seconds:
                        threading.Thread(target=call_api_async, args=(api_url,), daemon=True).start()
                        last_api_call = now

        # overlay cooldown status on frame
        with lock:
            remaining = max(0, int(cooldown_seconds - (time.time() - last_api_call)))
        if remaining > 0:
            text = f"Cooldown: {remaining}s"
            cv2.putText(annotated, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # cv2.imshow("Bird Detection (Pi4)", annotated)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
finally:
    cap.release()
    cv2.destroyAllWindows()
