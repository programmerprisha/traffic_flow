import cv2
from ultralytics import YOLO
import numpy as np
from typing import List, Dict


class VehicleDetector:
    def __init__(self):
        self.model = YOLO('yolov8m.pt')
        self.conf_threshold = 0.5
        # COCO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
        self.vehicle_classes = [2, 3, 5, 7]

    def detect(self, frame: np.ndarray) -> List[Dict]:
        result = self.model(frame, conf=self.conf_threshold,
                            classes=self.vehicle_classes, verbose=False)
        detections = []
        for box in result[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            detections.append({
                'bbox':       (x1, y1, x2, y2),
                'confidence': conf,
                'class_id':   cls,
                'center':     ((x1 + x2) // 2, (y1 + y2) // 2)
            })
        return detections

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self.model.names[det['class_id']]}: {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame