import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.detection.detector import VehicleDetector

print("Initializing detector...")
detector = VehicleDetector()

video_path = "data/raw/kaggle.mp4"
print(f"Opening video: {video_path}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Video not found at {video_path}")
    print("Please put a video file in data/raw/kaggle.mp4")
    exit(1)

print("Video opened! Press 'q' to quit.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break
    
    frame_count += 1
    
    # Process every 3rd frame for speed
    if frame_count % 3 == 0:
        detections = detector.detect(frame)
        annotated = detector.annotate_frame(frame.copy(), detections)
        
        # Show info
        cv2.putText(annotated, f"Vehicles: {len(detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Traffic Detection Test', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Processed {frame_count} frames")