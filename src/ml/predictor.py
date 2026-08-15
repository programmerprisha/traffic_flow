
import torch
import numpy as np
import json
import os

from src.ml.model import TrafficLSTM, DEVICE


def _level(score):
    if score < 25:  return "FREE FLOW",  (0, 200, 0)
    if score < 50:  return "LIGHT",      (0, 220, 100)
    if score < 70:  return "MODERATE",   (0, 165, 255)
    if score < 85:  return "HEAVY",      (0, 69, 255)
    return               "GRIDLOCK",     (0, 0, 220)


class CongestionPredictor:

    def __init__(self, model_dir="models/"):
        self.ready = False
        self.history = []
        self.last_prediction = None

        config_path = os.path.join(model_dir, 'config.json')
        weights_path = os.path.join(model_dir, 'model.pt')

        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            print(f"Model not found in {model_dir}. Train first.")
            return

        with open(config_path) as f:
            self.cfg = json.load(f)

        self.model = TrafficLSTM(
            num_features=self.cfg['num_features'],
            hidden_size=self.cfg['hidden_size'],
            num_layers=self.cfg['num_layers'],
            dropout=0.0   # disable dropout at inference
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        self.model.eval()

        self.feature_cols = self.cfg['feature_cols']
        self.mean = np.array(self.cfg['scaler_mean'], dtype=np.float32)
        self.std  = np.array(self.cfg['scaler_std'],  dtype=np.float32)
        self.window = self.cfg['window']
        self.lookahead = self.cfg['lookahead']

        self.ready = True
        print(f"LSTM predictor ready | "
              f"window={self.window}s | lookahead={self.lookahead}s | "
              f"val MAE={self.cfg.get('best_val_mae', '?'):.2f} pts")

    def update(self, zone_record: dict):
        """Feed one per-second record. Returns prediction or None if warming up."""
        self.history.append(zone_record)
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

        if len(self.history) < self.window or not self.ready:
            return None

        feat = np.array(
            [[row.get(c, 0.0) for c in self.feature_cols] for row in self.history],
            dtype=np.float32
        )
        feat = (feat - self.mean) / self.std
        x = torch.tensor(feat).unsqueeze(0).to(DEVICE)  # (1, T, F)

        with torch.no_grad():
            pred = self.model(x).item()

        self.last_prediction = max(0.0, min(100.0, pred))
        return self.last_prediction

    def draw_prediction(self, frame, current_score):
        import cv2
        h, w = frame.shape[:2]
        px, py = w - 290, 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 5, py - 5), (w - 5, py + 105), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "LSTM TRAFFIC PREDICTOR",
                    (px, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        cur_label, cur_color = _level(current_score)
        cv2.putText(frame, f"NOW  {current_score:>5.1f}/100  {cur_label}",
                    (px, py + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.44, cur_color, 1)

        pred = self.last_prediction
        if pred is not None:
            fut_label, fut_color = _level(pred)
            cv2.putText(frame, f"+{self.lookahead}s {pred:>5.1f}/100  {fut_label}",
                        (px, py + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, fut_color, 1)

            # trend arrow
            delta = pred - current_score
            arrow = "▲" if delta > 5 else ("▼" if delta < -5 else "●")
            arrow_color = (0, 0, 255) if delta > 5 else (0, 200, 0) if delta < -5 else (180, 180, 180)
            cv2.putText(frame, f"{arrow} {delta:+.1f} pts",
                        (px, py + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.44, arrow_color, 1)

            if pred > 70 and current_score < 45:
                cv2.putText(frame, "⚠ CONGESTION INCOMING",
                            (px, py + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 2)
        else:
            remaining = self.window - len(self.history)
            cv2.putText(frame, f"Warming up ({remaining}s remaining)...",
                        (px, py + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

        return frame