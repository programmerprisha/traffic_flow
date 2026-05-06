import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import random

from src.detection.detector import VehicleDetector
from src.detection.tracker import Tracker

print("Initializing...")

detector = VehicleDetector()
tracker = Tracker(max_distance=80, missing_frames_max=8)

video_path = "data/raw/kaggle.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video not found")
    exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"FPS:", fps)

track_colors = {}

def get_color(track_id):
    if track_id not in track_colors:
        random.seed(track_id)
        track_colors[track_id] = (
            random.randint(80, 255),
            random.randint(80, 255),
            random.randint(80, 255)
        )
        random.seed()
    return track_colors[track_id]


frame_count = 0
skip_frames = 2

last_tracks = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display = frame.copy()

    if frame_count % skip_frames == 0:

        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        detections = detector.detect(frame)
        last_tracks = tracker.update(detections, frame_count)

    for det in last_tracks:
        if 'track_id' not in det:
            continue

        x1, y1, x2, y2 = det['bbox']
        tid = det['track_id']

        if tid <= 0:
            continue

        color = get_color(tid)

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        cv2.putText(display, f"ID {tid}", (x1, y1 - 7),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        conf = det.get('confidence', 0)
        bar = int((x2 - x1) * conf)
        cv2.rectangle(display, (x1, y2), (x1 + bar, y2 + 3), (0, 255, 0), -1)


    active = sum(1 for t in tracker.tracks.values() if t['missing_frames'] < 5)

    cv2.putText(display, f"Vehicles: {len(last_tracks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(display, f"Active Tracks: {active}", (10, 55),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(display, f"Frame: {frame_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 
    cv2.imshow("Traffic Tracking (Stable)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif cv2.waitKey(1) & 0xFF == ord('r'):
        tracker = Tracker(max_distance=80, missing_frames_max=8)
        last_tracks = []
        print("Reset tracker!")


cap.release()
cv2.destroyAllWindows()

print("\nDone")
print("Frames:", frame_count)
print("Tracks:", len(tracker.tracks))
