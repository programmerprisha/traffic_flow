import cv2
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Dict
import yaml

class VehicleDetector: 
    def __init__(self, config_path: str = None): 
        ## prisha note = loading model 
        self.model = YOLO('yolov8m.pt')
        self.conf_threshold = 0.5
        ## prisha note = the classes I want to detect are car, bus, truck, motorcycle
        self.vehicle_classes = [2, 5, 7, 3]

    def detect(self, frame: np.ndarray) -> List[Dict]: 
        ## prisha note = detecting objects in the frame, processes and returns objects containing coordinates
        result = self.model(frame, conf=self.conf_threshold, classes=self.vehicle_classes, verbose=False)

        detections = []
        for box in result[0].boxes: 
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                    'bbox': (x1, y1, x2, y2), 
                    'confidence': conf, 
                    'class_id': cls, 
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2) 
    })

        return detections

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray: 
        for det in detections: 
            x1, y1, x2, y2 = det['bbox']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame


        