import cv2
import numpy as np
from typing import List, Dict, Tuple
from collections import deque

class Tracker: 
    def __init__(self, max_distance=100, missing_frames_max=10): 
        self.next_id = 1
        self.tracks = {}  
        self.max_distance = max_distance
        self.missing_frames_max = missing_frames_max

    def update(self, detections, frame_idx):
        if not detections: 
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['missing_frames'] += 1
                ## prisha note = if a track has been missing for too long, we remove it
                if self.tracks[track_id]['missing_frames'] > self.missing_frames_max:
                    del self.tracks[track_id]
            return []
        
        detection_centers = [d['center'] for d in detections]

        if not self.tracks:
            for i, det in enumerate(detections):
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'positions' : deque(maxlen=30),
                    'last_center': det['center'],
                    'last_bbox': det['bbox'],
                    'missing_frames': 0, 
                    'first_frame': frame_idx
                }
                self.tracks[track_id]['positions'].append(det['center'])
                detections[i]['track_id'] = track_id
            return detections
        
        for track_id in self.tracks:
            self.tracks[track_id]['missing_frames'] += 1

        all_matches = []

        for i, det in enumerate(detections): 
            for track_id, track in self.tracks.items():

                dist = np.sqrt((det['center'][0] - track['last_center'][0]) ** 2 + (det['center'][1] - track['last_center'][1]) ** 2)

                det_width = det['bbox'][2] - det['bbox'][0]
                det_height = det['bbox'][3] - det['bbox'][1]
                track_width = track['last_bbox'][2] - track['last_bbox'][0]
                track_height = track['last_bbox'][3] - track['last_bbox'][1]
                size_diff = abs(det_width - track_width) + abs(det_height - track_height)

                score = dist + size_diff * 0.1

                if score < self.max_distance:
                    all_matches.append((score, track_id, i))
            
        all_matches.sort(key=lambda x: x[0])

        used_tracks = set()
        used_detections = set()
        matched = []

        
        for score, track_id, det_idx in all_matches: 
            if track_id in used_tracks or det_idx in used_detections:
                continue

            matched.append((track_id, det_idx))
            used_tracks.add(track_id)
            used_detections.add(det_idx)


        for track_id, det_idx in matched: 
            track = self.tracks[track_id]
            det = detections[det_idx]
            track['positions'].append(det['center'])
            track['last_center'] = det['center']
            track['last_bbox'] = det['bbox']
            track['missing_frames'] = 0
            detections[det_idx]['track_id'] = track_id

        for i, det in enumerate(detections): 
            if i not in used_detections:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'positions': deque(maxlen=30),
                    'last_center': det['center'],
                    'last_bbox': det['bbox'],
                    'missing_frames': 0,
                    'first_frame': frame_idx
                }
                self.tracks[track_id]['positions'].append(det['center'])
                detections[i]['track_id'] = track_id
    

        for track_id, track in self.tracks.items():
            if len(track['positions']) >= 3: 
                positions = list(track['positions'])

                avg_x = int(np.mean([p[0] for p in positions[-3:]]))
                avg_y = int(np.mean([p[1] for p in positions[-3:]]))
                track['smooth_center'] = (avg_x, avg_y)

        return detections

    def get_track_history(self, track_id):
        if track_id in self.tracks:
            return list(self.tracks[track_id]['positions'])
        return []
    
    def get_track_speed(self, track_id, fps=30, pixels_per_meter=0.03): 
        positions = self.get_track_history(track_id)

        if len(positions) < 10: 
            return 0


        frames_to_check = min(fps, len(positions))
        
        start_pos = positions[-frames_to_check]
        end_pos = positions[-1]

        pixel_distance = np.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)
        meters = pixel_distance * pixels_per_meter
        seconds = frames_to_check / fps

        if seconds == 0:
            return 0
        speed_mps = meters / seconds
        speed_kmph = speed_mps * 3.6
        return speed_kmph
            