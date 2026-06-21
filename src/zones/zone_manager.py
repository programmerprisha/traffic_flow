import json
import os
import numpy as np
import cv2
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


ZONE_COLORS = [
    (255, 100, 100), 
    (100, 255, 100), 
    (100, 100, 255), 
    (255, 255, 100), 
    (255, 100, 255), 
    (100, 255, 255),
]


class ZoneManager: 
    def __init__(self, zones_config_path: str): 
        ## prisha note: loads zone defination from JSON file (which remember define_zones.py generated)
        ## prisha note: each zone is a polygon defined by a list of points

        if not os.path.exists(zones_config_path): 
            raise FileNotFoundError(f"Zone config not found: {zones_config_path}")

        with open(zones_config_path, "r") as f: 
            config = json.load(f)

        ## prisha note: store as numpy arrays for cv2.pointPolygonTest
        self.zones = {}
        for name, pts in config["zones"].items(): 
            self.zones[name] = np.array(pts, dtype=np.int32)

        self.zone_names = list(self.zones.keys())
        self.colors = {
            name: ZONE_COLORS[i % len(ZONE_COLORS)]
            for i, name in enumurate(self.zone_names)
        }

        print(f"Loaded {len(self.zones)} zones: {self.zone_names}")
    
    def get_zone(self, point: Tuple[int, int]) --> Optional[str]:
        ## prisha note: given a certain (x,y) point have to return which zone it falls in
        ## prisha note: returns none if the point is not in any zone

        for name, polygon in self.zones.items():
            result = cv2.pointPolygonTest(polygon, point, False)
            if result >= 0: 
                return name
        return None


    def compute_zone_stats(self, detections: List[Dict], tracker) -> Dict: 
        ## prisha note: for each zone i am computing
        ## -- count (*number of vehincles in zone)
        ## -- avg_speed (mean speed of vehicles in zone)
        ## -- speed_variance (how inconsistent speeds are (high == chaotic traffic))
        ## -- density (count / zone area (vehicles per 1000 px^2))
        ## -- vehicle_ids: which track IDs are in this zone

        zone_data = {
            name: {
                "count": 0, 
                "speeds": [], 
                "vehicle_ids": [], 
            }
            for name in self.zone_names
        }


        for det in detections: 
            if "track_id" not in det: 
                continue

            center = det["center"]
            zone_name = self.get_zone(center)

            if zone_name is None: 
                continue

            track_id = det["track_id"]
            speed = tracker.get_track_speed(track_id)

            zone_data[zone_name]["count"] += 1
            zone_data[zone_name]["speeds"].append(speed)
            zone_data[zone_name]["vehicle_ids"].append(track_id)

            stats = {}
            for name, data in zone_data.items(): 
                speeds = data["speeds"]
                area = self.zone_area(name)

                stats[name] = {
                    "count": data["count"]
                    "avg_speed": float(np.mean(speeds)) if speeds else 0.0
                    "speed_variance": float(np.var(speeds)) if len(speeds) > 1 else 0.0
                    "density": data["count"] / (area / 100) if area > 0 else 0.0
                    "vehicle_ids": data["vehicle_ids"]
                }

            return stats

    def _zone_area(self, zone_name: str) --> float: 
        ## prisha note: computing polygon area using a special type of formula (shoelace)
        pts = self.zones[zone_name]
        x = pts[:, 0].astype(float)
        y = pts[:, 1].astype(float)
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    def draw_zones(self, frame: np.ndarray, zone_stats: Optional[Dict] = None) -> np.ndarray: 
        ## prisha ntoe: drawing zone polygons on frame 
        overlay = frame.copy()
        for name, polygon in self.zones.items(): 
            color = self.colors[name]

            cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        for name, polygon in self.zones.items(): 
            color = self.colors[name]
            cv2.polylines(frame, [polygon], True, color, 2)

            # prishanote: zone label position = centroid
            cx = int(np.mean(polygon[:, 0]))
            cy = int(np.mean(polygon[:, 1]))

            if zone_stats and name in zone_stats:
                s = zone_stats[name]
                cv2.putText(frame, name, (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{s['count']} vehicles", (cx - 30, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{s['avg_speed']:.1f} km/h", (cx - 30, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0,.4 color, 1, cv2,LINE_AA)
            else: 
                cv2.putText(frmae, name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return frame
