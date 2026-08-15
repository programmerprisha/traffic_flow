
import torch
import torch.nn as nn
import numpy as np
import os
import json
import argparse

from src.ml.model import TrafficDataset, TrafficLSTM, WINDOW, LOOKAHEAD, DEVICE


def time_split(dataset, train_ratio=0.8):
    """Split preserving temporal order — NO random shuffle."""
    n = len(dataset)
    train_n = int(n * train_ratio)
    train_set = torch.utils.data.Subset(dataset, range(train_n))
    val_set   = torch.utils.data.Subset(dataset, range(train_n, n))
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")
    return train_set, val_set


def train(csv_path, output_dir, epochs=100, batch_size=32, lr=1e-3, patience=15):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nDevice: {DEVICE}")

    # ── Data ──────────────────────────────────────────────────────
    full_dataset = TrafficDataset(csv_path, window=WINDOW, lookahead=LOOKAHEAD,
                                  fit_scaler=True)
    train_set, val_set = time_split(full_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False  # NO shuffle — preserve time order
    )
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────
    num_features = full_dataset.X.shape[2]
    model = TrafficLSTM(num_features=num_features, hidden_size=128,
                        num_layers=2, dropout=0.3).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()
    mae_fn = nn.L1Loss()

    # ── Training loop ─────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    print(f"\nTraining for up to {epochs} epochs (early stop patience={patience})...")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # validate
        model.eval()
        val_losses, val_maes = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())
                val_maes.append(mae_fn(pred, y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_mae    = np.mean(val_maes)

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{epochs}  "
                  f"train_loss={train_loss:.3f}  "
                  f"val_loss={val_loss:.3f}  "
                  f"val_MAE={val_mae:.2f} pts  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # save best weights
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_weights.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Best val MAE : {min(history['val_mae']):.2f} congestion points")

    # ── Save everything ───────────────────────────────────────────
    # load best weights back
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_weights.pt'),
                                     map_location=DEVICE))
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

    # save config (needed to rebuild model at inference time)
    config = {
        'num_features': num_features,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'window': WINDOW,
        'lookahead': LOOKAHEAD,
        'feature_cols': full_dataset.feature_cols,
        'scaler_mean': full_dataset.scaler['mean'].tolist(),
        'scaler_std': full_dataset.scaler['std'].tolist(),
        'best_val_mae': float(min(history['val_mae'])),
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  model.pt     — weights")
    print(f"  config.json  — architecture + scaler")
    print(f"  history.json — loss curves")
    print(f"\nNext: python main.py --model {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default='data/stats/zone_stats.csv')
    parser.add_argument('--output',  default='models/')
    parser.add_argument('--epochs',  type=int, default=100)
    parser.add_argument('--batch',   type=int, default=32)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--patience',type=int, default=15)
    args = parser.parse_args()
    train(args.input, args.output, args.epochs, args.batch, args.lr, args.patience)