import numpy as np
import csv
import os
from collections import defaultdict, deque


# COCO class IDs used by YOLOv8
CLASS_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


class ZoneAnalyzer:

    def __init__(self, frame_width, frame_height, num_zones=4, fps=30):
        self.W = frame_width
        self.H = frame_height
        self.num_zones = num_zones
        self.fps = fps
        self.zone_height = frame_height // num_zones

        self.zones = {
            i: {
                'y1': i * self.zone_height,
                'y2': (i + 1) * self.zone_height if i < num_zones - 1 else frame_height
            }
            for i in range(num_zones)
        }

        # per-frame buffer, flushed every second
        self.frame_buffer = []
        self.prev_zone_assignment = {}   # track_id → zone_id (for entry/exit)

        self.stats_history = []
        self.second_index = 0

    def get_zone(self, cy):
        for zid, z in self.zones.items():
            if z['y1'] <= cy < z['y2']:
                return zid
        return self.num_zones - 1

    def update(self, detections, tracker, frame_idx):
        """Call once per frame."""
        zone_snap = defaultdict(list)
        curr_zone_assignment = {}

        for det in detections:
            if 'track_id' not in det:
                continue
            cx, cy = det['center']
            zid = self.get_zone(cy)
            curr_zone_assignment[det['track_id']] = zid

            x1, y1, x2, y2 = det['bbox']
            bbox_area = (x2 - x1) * (y2 - y1)

            zone_snap[zid].append({
                'track_id': det['track_id'],
                'speed': tracker.get_track_speed(det['track_id'], fps=self.fps),
                'class_id': det.get('class_id', 2),
                'cx': cx, 'cy': cy,
                'bbox_area': bbox_area,
                'confidence': det.get('confidence', 1.0),
            })

        # compute zone entry/exit counts
        entries = defaultdict(int)
        exits = defaultdict(int)
        for tid, new_z in curr_zone_assignment.items():
            old_z = self.prev_zone_assignment.get(tid)
            if old_z is not None and old_z != new_z:
                exits[old_z] += 1
                entries[new_z] += 1
        self.prev_zone_assignment = curr_zone_assignment

        self.frame_buffer.append({'zones': zone_snap, 'entries': entries, 'exits': exits})

        if len(self.frame_buffer) >= self.fps:
            self._flush_second()

        return zone_snap

    def _flush_second(self):
        record = {'second': self.second_index}

        total_count_all = 0
        all_speeds = []

        for zid in range(self.num_zones):
            # aggregate over the second's frames
            all_ids = set()
            speeds = []
            class_counts = defaultdict(int)
            vy_sum = 0.0
            bbox_area_sum = 0.0
            entry_sum = 0
            exit_sum = 0

            zone_area = self.W * (self.zones[zid]['y2'] - self.zones[zid]['y1'])

            for frame_snap in self.frame_buffer:
                for v in frame_snap['zones'][zid]:
                    all_ids.add(v['track_id'])
                    if v['speed'] > 0:
                        speeds.append(v['speed'])
                    class_counts[v['class_id']] += 1
                    vy_sum += v['cy']
                    bbox_area_sum += v['bbox_area']
                entry_sum += frame_snap['entries'].get(zid, 0)
                exit_sum += frame_snap['exits'].get(zid, 0)

            count = len(all_ids)
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            speed_std = float(np.std(speeds)) if len(speeds) > 1 else 0.0
            density = count / (self.zone_height / 100)
            occupancy = min(bbox_area_sum / (zone_area * self.fps + 1e-6), 1.0)

            total_vehicles = sum(class_counts.values()) or 1
            car_ratio   = class_counts.get(2, 0) / total_vehicles
            truck_ratio = class_counts.get(7, 0) / total_vehicles
            bus_ratio   = class_counts.get(5, 0) / total_vehicles
            moto_ratio  = class_counts.get(3, 0) / total_vehicles

            # flow direction: positive = moving toward bottom (zone increases)
            flow_dir = 0.0
            if count > 0 and len(self.frame_buffer) > 1:
                first_snap = [v['cy'] for v in self.frame_buffer[0]['zones'][zid]]
                last_snap  = [v['cy'] for v in self.frame_buffer[-1]['zones'][zid]]
                if first_snap and last_snap:
                    flow_dir = float(np.mean(last_snap) - np.mean(first_snap))

            z = f'z{zid}'
            record[f'{z}_count']      = count
            record[f'{z}_avg_speed']  = round(avg_speed, 2)
            record[f'{z}_speed_std']  = round(speed_std, 2)
            record[f'{z}_density']    = round(density, 2)
            record[f'{z}_occupancy']  = round(float(occupancy), 4)
            record[f'{z}_car_ratio']  = round(car_ratio, 3)
            record[f'{z}_truck_ratio']= round(truck_ratio, 3)
            record[f'{z}_bus_ratio']  = round(bus_ratio, 3)
            record[f'{z}_moto_ratio'] = round(moto_ratio, 3)
            record[f'{z}_flow_dir']   = round(flow_dir, 2)
            record[f'{z}_entries']    = entry_sum
            record[f'{z}_exits']      = exit_sum

            total_count_all += count
            all_speeds.extend(speeds)

        # global features
        record['total_count']       = total_count_all
        record['global_avg_speed']  = round(float(np.mean(all_speeds)) if all_speeds else 0.0, 2)
        record['global_speed_std']  = round(float(np.std(all_speeds)) if len(all_speeds) > 1 else 0.0, 2)
        record['congestion_score']  = self._congestion_score(record)
        record['congestion_label']  = self._congestion_label(record['congestion_score'])

        self.stats_history.append(record)
        self.frame_buffer = []
        self.second_index += 1

    def _congestion_score(self, r):
        """
        Multi-factor congestion score 0–100.
        Combines: vehicle count, speed, occupancy, density.
        """
        count_norm  = min(r['total_count'] / 40, 1.0)
        speed       = r['global_avg_speed']
        speed_norm  = 1.0 - min(speed / 80.0, 1.0)   # low speed = high congestion
        occupancy   = np.mean([r.get(f'z{i}_occupancy', 0) for i in range(self.num_zones)])
        density     = np.mean([r.get(f'z{i}_density', 0) for i in range(self.num_zones)])
        density_norm = min(density / 10.0, 1.0)

        score = (count_norm * 30 + speed_norm * 40 +
                 occupancy * 20 + density_norm * 10)
        return round(float(score), 2)

    def _congestion_label(self, score):
        if score < 25:  return 0  # free flow
        if score < 50:  return 1  # light
        if score < 70:  return 2  # moderate
        if score < 85:  return 3  # heavy
        return 4                  # gridlock

    def save_stats(self, path):
        if self.frame_buffer:
            self._flush_second()
        if not self.stats_history:
            print("No stats to save.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.stats_history[0].keys()))
            writer.writeheader()
            writer.writerows(self.stats_history)
        print(f"Saved {len(self.stats_history)} seconds → {path}")
        print(f"  Features per timestep: {len(self.stats_history[0])} columns")

    def draw_zones(self, frame):
        import cv2
        for zid, z in self.zones.items():
            cv2.line(frame, (0, z['y1']), (self.W, z['y1']), (80, 80, 220), 1)
            cv2.putText(frame, f"Z{zid}", (8, z['y1'] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 220), 1)
        return frame

    @property
    def feature_columns(self):
        """All feature column names (excludes second, labels)."""
        if not self.stats_history:
            return []
        exclude = {'second', 'congestion_score', 'congestion_label'}
        return [k for k in self.stats_history[0].keys() if k not in exclude]

    @property
    def num_features(self):
        return len(self.feature_columns)