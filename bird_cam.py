import time
import urllib.request
import urllib.error

import cv2
from ultralytics import YOLO
import numpy as np

# Use a stronger YOLO model and larger input size for better bird detection quality.
model = YOLO("yolov8m.pt", task="detect")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Request automatic focus if the webcam supports it.
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

api_url = "https://agri-sound.vercel.app/api/play"
last_api_call = 0.0
cooldown_seconds = 30.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Directly detect birds on the current frame with one display window.
    results = model(frame, imgsz=640, conf=0.30, iou=0.45, classes=[14], verbose=False)
    annotated = results[0].plot() if len(results) > 0 else frame

    # If a bird is detected, call the API once and then wait 30 seconds before calling again.
    if len(results) > 0 and len(results[0].boxes) > 0:
        now = time.time()
        if now - last_api_call >= cooldown_seconds:
            try:
                with urllib.request.urlopen(api_url, timeout=5) as response:
                    response.read()
                print(f"Bird detected: called API at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}")
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                print(f"API call failed: {exc}")
            last_api_call = now

    cv2.imshow("Bird Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
