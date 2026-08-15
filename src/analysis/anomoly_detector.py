import numpy as np
import os
from collections import deque
from datetime import timedelta


class AnomalyDetector:

    def __init__(self, fps=30, baseline_window=30):
        self.fps = fps
        self.baseline_window = baseline_window  # seconds for rolling baseline
        self.history = deque(maxlen=baseline_window)
        self.incidents = []

    def _zscore(self, value, key):
        """Z-score of value vs rolling baseline for this key."""
        vals = [r[key] for r in self.history if key in r]
        if len(vals) < 5:
            return 0.0
        mu, sigma = np.mean(vals), np.std(vals)
        return (value - mu) / (sigma + 1e-8)

    def update(self, record: dict) -> list:
        anomalies = []
        t = int(record.get('second', 0))

        if len(self.history) >= 5:
            # 1. Congestion spike (z-score > 2.5)
            z = self._zscore(record['congestion_score'], 'congestion_score')
            if z > 2.5:
                ev = {
                    'type': 'CONGESTION_SPIKE',
                    'second': t,
                    'detail': f"Congestion {record['congestion_score']:.1f} is {z:.1f}σ above baseline",
                    'severity': 'HIGH' if record['congestion_score'] > 70 else 'MEDIUM'
                }
                anomalies.append(ev)
                self.incidents.append(ev)

            # 2. Speed collapse (z-score < -2.5)
            z_speed = self._zscore(record.get('global_avg_speed', 0), 'global_avg_speed')
            if z_speed < -2.5 and record.get('global_avg_speed', 100) < 30:
                ev = {
                    'type': 'SPEED_COLLAPSE',
                    'second': t,
                    'detail': f"Speed {record.get('global_avg_speed', 0):.1f} km/h is {abs(z_speed):.1f}σ below baseline",
                    'severity': 'HIGH'
                }
                anomalies.append(ev)
                self.incidents.append(ev)

            # 3. Sudden density spike in any zone
            for i in range(4):
                key = f'z{i}_density'
                if key in record:
                    z_d = self._zscore(record[key], key)
                    if z_d > 3.0:
                        ev = {
                            'type': 'DENSITY_SPIKE',
                            'second': t,
                            'detail': f"Zone {i} density {record[key]:.2f} is {z_d:.1f}σ above baseline",
                            'severity': 'MEDIUM'
                        }
                        anomalies.append(ev)
                        self.incidents.append(ev)

            # 4. High speed std = erratic driving
            z_std = self._zscore(record.get('global_speed_std', 0), 'global_speed_std')
            if z_std > 2.5 and record.get('global_speed_std', 0) > 20:
                ev = {
                    'type': 'ERRATIC_SPEEDS',
                    'second': t,
                    'detail': f"Speed std dev {record.get('global_speed_std', 0):.1f} km/h — possible incident",
                    'severity': 'MEDIUM'
                }
                anomalies.append(ev)
                self.incidents.append(ev)

        self.history.append(record)
        return anomalies

    def draw_anomalies(self, frame, anomalies):
        import cv2
        if not anomalies:
            return frame
        h = frame.shape[0]
        for i, ev in enumerate(anomalies[:3]):
            color = (0, 0, 255) if ev['severity'] == 'HIGH' else (0, 140, 255)
            cv2.putText(frame, f"⚠ {ev['type']}",
                        (10, h - 60 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
        return frame

    def generate_report(self, output_path, video_path="unknown", total_seconds=0):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        high = sum(1 for i in self.incidents if i['severity'] == 'HIGH')
        med  = sum(1 for i in self.incidents if i['severity'] == 'MEDIUM')

        lines = [
            "=" * 60,
            "TRAFFIC FLOW — INCIDENT REPORT",
            "=" * 60,
            f"Video    : {video_path}",
            f"Duration : {timedelta(seconds=total_seconds)}",
            f"Events   : {len(self.incidents)} total ({high} HIGH, {med} MEDIUM)",
            "=" * 60, "",
        ]

        if not self.incidents:
            lines.append("No anomalies detected.")
        else:
            by_type = {}
            for inc in self.incidents:
                by_type.setdefault(inc['type'], []).append(inc)
            for etype, evs in by_type.items():
                lines.append(f"── {etype} ({len(evs)}) ──")
                for e in evs:
                    lines.append(f"  [{timedelta(seconds=e['second'])}] "
                                 f"[{e['severity']}] {e['detail']}")
                lines.append("")

        lines += ["=" * 60, "END OF REPORT", "=" * 60]
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"Incident report → {output_path}")