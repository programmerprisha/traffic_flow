
import torch
import torch.nn as nn
import numpy as np
import csv
import os

WINDOW = 30        # seconds of history as input
LOOKAHEAD = 60     # seconds ahead to predict
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'   # Apple Silicon GPU
else:
    DEVICE = 'cpu'


# ── Dataset ──────────────────────────────────────────────────────────────────

class TrafficDataset(torch.utils.data.Dataset):
    """
    Sliding window over the zone stats CSV.
    Each sample: (window of features, target congestion score T+lookahead)
    """

    def __init__(self, csv_path, window=WINDOW, lookahead=LOOKAHEAD,
                 feature_cols=None, scaler=None, fit_scaler=False):
        rows = self._load(csv_path)
        if len(rows) < window + lookahead + 5:
            raise ValueError(
                f"Need ≥{window + lookahead + 5} seconds of data, got {len(rows)}. "
                "Process a longer video."
            )

        self.feature_cols = feature_cols or self._infer_features(rows)
        raw = np.array([[r[c] for c in self.feature_cols] for r in rows], dtype=np.float32)
        targets = np.array([r['congestion_score'] for r in rows], dtype=np.float32)

        # z-score normalisation — critical for LSTM stability
        if fit_scaler:
            self.mean = raw.mean(axis=0)
            self.std  = raw.std(axis=0) + 1e-8
        else:
            self.mean = scaler['mean']
            self.std  = scaler['std']

        self.scaler = {'mean': self.mean, 'std': self.std}
        raw = (raw - self.mean) / self.std

        self.X, self.y = [], []
        for i in range(window, len(rows) - lookahead):
            self.X.append(raw[i - window:i])          # (window, features)
            self.y.append(targets[i + lookahead])      # scalar

        self.X = np.array(self.X, dtype=np.float32)   # (N, window, features)
        self.y = np.array(self.y, dtype=np.float32)   # (N,)

        print(f"Dataset: {len(self.X)} samples | "
              f"window={window}s | lookahead={lookahead}s | "
              f"features={len(self.feature_cols)}")

    def _load(self, path):
        with open(path) as f:
            reader = csv.DictReader(f)
            return [{k: float(v) for k, v in row.items()} for row in reader]

    def _infer_features(self, rows):
        exclude = {'second', 'congestion_label'}
        return [k for k in rows[0].keys() if k not in exclude]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ── Model ────────────────────────────────────────────────────────────────────

class TrafficLSTM(nn.Module):
    """
    Stacked 2-layer LSTM with dropout + LayerNorm.

    Shapes flowing through:
      x:              (B, T, F)       B=batch, T=window, F=features
      lstm1 out:      (B, T, hidden)
      lstm2 out:      (B, T, hidden)
      last hidden:    (B, hidden)     take last timestep
      fc head:        (B, 1)
    """

    def __init__(self, num_features, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.input_norm = nn.LayerNorm(num_features)

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

        # regression head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_norm(x)
        out, _ = self.lstm(x)           # out: (B, T, hidden)
        last = out[:, -1, :]            # take last timestep: (B, hidden)
        last = self.dropout(last)
        last = self.norm(last)
        return self.fc(last).squeeze(-1)  # (B,)