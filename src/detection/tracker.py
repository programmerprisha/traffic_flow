import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment


# ── Kalman Filter ────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Models one vehicle's state as [x, y, w, h, vx, vy, vw, vh]
    where (x,y) is bbox center, (w,h) is size, v* are velocities.

    Prediction step: x_t = F * x_{t-1}  (constant velocity model)
    Update step:     x_t = x_t + K * (z_t - H * x_t)  (correct with detection)
    """

    count = 0  # global ID counter

    def __init__(self, bbox):
        # state: [cx, cy, w, h, vx, vy, vw, vh]
        self.kf_dim_x = 8
        self.kf_dim_z = 4  # we observe [cx, cy, w, h]

        # state transition matrix (constant velocity)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # position += velocity

        # measurement matrix (we only observe position/size, not velocity)
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        # process noise — how much we trust the motion model
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01  # velocities change slowly

        # measurement noise — how much we trust detections
        self.R = np.eye(4)
        self.R[:2, :2] *= 1.0   # position noise
        self.R[2:, 2:] *= 10.0  # size noise (detectors vary more in size)

        # initial covariance — high uncertainty in velocities at start
        self.P = np.eye(8)
        self.P[4:, 4:] *= 1000.0

        # state vector
        cx, cy, w, h = self._bbox_to_center(bbox)
        self.x = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=float)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = deque(maxlen=60)   # position history
        self.speed_history = deque(maxlen=30)
        self.hit_streak = 0
        self.age = 0
        self.class_id = None

    def _bbox_to_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    def _center_to_bbox(self):
        cx, cy, w, h = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        return (int(cx - w / 2), int(cy - h / 2),
                int(cx + w / 2), int(cy + h / 2))

    def predict(self):
        """Kalman predict step — where do we expect this vehicle to be?"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self._center_to_bbox()

    def update(self, bbox, class_id=None):
        """Kalman update step — correct prediction with actual detection."""
        cx, cy, w, h = self._bbox_to_center(bbox)
        z = np.array([[cx], [cy], [w], [h]])

        # innovation
        y = z - self.H @ self.x

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # update state
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hit_streak += 1
        self.history.append((int(cx), int(cy)))
        if class_id is not None:
            self.class_id = class_id

    def get_center(self):
        return (int(self.x[0, 0]), int(self.x[1, 0]))

    def get_velocity(self):
        """Pixel velocity (vx, vy) from Kalman state."""
        return float(self.x[4, 0]), float(self.x[5, 0])

    def get_speed_kmh(self, fps=30, pixels_per_meter=0.05):
        vx, vy = self.get_velocity()
        pixel_speed = np.sqrt(vx**2 + vy**2)  # pixels/frame
        mps = pixel_speed * pixels_per_meter * fps
        return mps * 3.6

    def get_smooth_center(self):
        if len(self.history) >= 3:
            recent = list(self.history)[-3:]
            return (int(np.mean([p[0] for p in recent])),
                    int(np.mean([p[1] for p in recent])))
        return self.get_center()


# ── IoU helper ───────────────────────────────────────────────────────────────

def compute_iou(bb1, bb2):
    """Intersection over Union between two bboxes [x1,y1,x2,y2]."""
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(trackers_pred, detections):
    """Build IoU cost matrix: rows=trackers, cols=detections."""
    mat = np.zeros((len(trackers_pred), len(detections)))
    for t, tb in enumerate(trackers_pred):
        for d, det in enumerate(detections):
            mat[t, d] = compute_iou(tb, det['bbox'])
    return mat


# ── Main Tracker ─────────────────────────────────────────────────────────────

class Tracker:
    """
    SORT-style tracker:
      1. Predict all existing tracks forward one step (Kalman)
      2. Build IoU matrix between predicted boxes and new detections
      3. Hungarian algorithm finds optimal assignment
      4. Update matched tracks, create new ones, delete lost ones
    """

    def __init__(self, iou_threshold=0.25, max_age=10, min_hits=2):
        self.iou_threshold = iou_threshold  # min IoU to consider a match
        self.max_age = max_age              # frames before dropping a lost track
        self.min_hits = min_hits            # frames before confirming a new track
        self.trackers = []                  # list of KalmanBoxTracker
        KalmanBoxTracker.count = 0

    def update(self, detections, frame_idx):
        # ── Step 1: predict ──────────────────────────────────────
        predicted_boxes = []
        for t in self.trackers:
            pred = t.predict()
            predicted_boxes.append(pred)

        # ── Step 2: IoU matrix ───────────────────────────────────
        matched_t = set()
        matched_d = set()

        if self.trackers and detections:
            iou_mat = iou_matrix(predicted_boxes, detections)

            # Hungarian: maximise IoU → minimise negative IoU
            row_idx, col_idx = linear_sum_assignment(-iou_mat)

            for t_idx, d_idx in zip(row_idx, col_idx):
                if iou_mat[t_idx, d_idx] >= self.iou_threshold:
                    matched_t.add(t_idx)
                    matched_d.add(d_idx)
                    self.trackers[t_idx].update(
                        detections[d_idx]['bbox'],
                        detections[d_idx].get('class_id')
                    )
                    detections[d_idx]['track_id'] = self.trackers[t_idx].id

        # ── Step 3: new detections → new tracks ─────────────────
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_d:
                trk = KalmanBoxTracker(det['bbox'])
                trk.class_id = det.get('class_id')
                self.trackers.append(trk)
                det['track_id'] = trk.id

        # ── Step 4: remove dead tracks ───────────────────────────
        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.max_age]

        return detections

    def get_track_speed(self, track_id, fps=30, pixels_per_meter=0.05):
        for t in self.trackers:
            if t.id == track_id:
                return t.get_speed_kmh(fps=fps, pixels_per_meter=pixels_per_meter)
        return 0.0

    def get_track_history(self, track_id):
        for t in self.trackers:
            if t.id == track_id:
                return list(t.history)
        return []

    def get_active_count(self):
        return sum(1 for t in self.trackers
                   if t.time_since_update == 0 and t.hit_streak >= self.min_hits)