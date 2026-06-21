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
    