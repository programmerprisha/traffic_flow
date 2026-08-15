
import cv2
import os
import argparse
import numpy as np

from src.detection.detector import VehicleDetector
from src.detection.tracker import Tracker
from src.analysis.zone_analyzer import ZoneAnalyzer
from src.analysis.anomaly_detector import AnomalyDetector


def load_predictor(model_dir):
    if not model_dir:
        return None
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"No model found in {model_dir} — running without predictions.")
        return None
    from src.ml.predictor import CongestionPredictor
    return CongestionPredictor(model_dir)


def draw_hud(frame, frame_idx, fps, total_seen, congestion_score,
             anomalies, predictor, zone_analyzer):
    """Top-left stats HUD."""
    second = frame_idx // fps

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, 110), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"t={second}s  |  frame {frame_idx}",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.putText(frame, f"Vehicles tracked: {total_seen}",
                (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(frame, f"Active now: {zone_analyzer.num_features} features/step",
                (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    # congestion bar
    score = congestion_score
    bar_w = int((score / 100) * 220)
    color = (0, 200, 0) if score < 30 else (0, 165, 255) if score < 60 else (0, 0, 220)
    cv2.rectangle(frame, (8, 62), (228, 76), (50, 50, 50), -1)
    cv2.rectangle(frame, (8, 62), (8 + bar_w, 76), color, -1)
    cv2.putText(frame, f"Congestion: {score:.1f}/100",
                (8, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # LSTM prediction panel
    if predictor:
        predictor.draw_prediction(frame, congestion_score)

    # anomaly warnings
    if anomalies:
        dummy = AnomalyDetector()
        dummy.draw_anomalies(frame, anomalies)

    return frame


def main(video_path, output_path, stats_path, model_dir=None, report_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps}fps  (~{total} frames / ~{total//fps}s)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector  = VehicleDetector()
    tracker   = Tracker(iou_threshold=0.25, max_age=10, min_hits=2)
    zones     = ZoneAnalyzer(width, height, num_zones=4, fps=fps)
    anomalies = AnomalyDetector(fps=fps)
    predictor = load_predictor(model_dir)

    frame_idx = 0
    total_seen = set()
    current_congestion = 0.0
    current_anomalies = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Detect + Track ───────────────────────────────────────
        detections = detector.detect(frame)
        detections = tracker.update(detections, frame_idx)
        annotated  = detector.annotate_frame(frame, detections)

        for det in detections:
            if 'track_id' not in det:
                continue
            total_seen.add(det['track_id'])
            x1, y1, x2, y2 = det['bbox']
            speed = tracker.get_track_speed(det['track_id'], fps=fps)
            cv2.putText(annotated, f"#{det['track_id']} {speed:.0f}km/h",
                        (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

        # ── Zone analysis ────────────────────────────────────────
        zones.update(detections, tracker, frame_idx)
        annotated = zones.draw_zones(annotated)

        # ── Per-second updates ───────────────────────────────────
        if frame_idx % fps == 0 and zones.stats_history:
            record = zones.stats_history[-1]
            current_congestion = record['congestion_score']

            if predictor:
                predictor.update(record)

            current_anomalies = anomalies.update(record)

        # ── HUD ──────────────────────────────────────────────────
        annotated = draw_hud(annotated, frame_idx, fps, len(total_seen),
                             current_congestion, current_anomalies, predictor, zones)

        out.write(annotated)
        frame_idx += 1

        if frame_idx % (fps * 15) == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"  {frame_idx} frames ({pct:.0f}%) — congestion: {current_congestion:.1f}/100")

    cap.release()
    out.release()

    zones.save_stats(stats_path)

    if report_path:
        anomalies.generate_report(report_path, video_path, frame_idx // fps)

    print(f"\nDone.")
    print(f"  Annotated video  : {output_path}")
    print(f"  Zone stats CSV   : {stats_path}  ({zones.num_features} features/timestep)")
    if report_path:
        print(f"  Incident report  : {report_path}")
    print(f"\nNext — train LSTM:")
    print(f"  pip install torch")
    print(f"  python -m src.ml.train --input {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default='data/raw/kaggle.mp4')
    parser.add_argument('--output',  default='data/annotated/annotated_kaggle.mp4')
    parser.add_argument('--stats',   default='data/stats/zone_stats.csv')
    parser.add_argument('--model',   default=None,
                        help='Path to models/ dir. Omit until after training.')
    parser.add_argument('--report',  default='data/stats/incident_report.txt')
    args = parser.parse_args()
    main(args.input, args.output, args.stats, args.model, args.report)