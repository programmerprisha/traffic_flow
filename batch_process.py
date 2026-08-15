import os
import cv2
import csv
import argparse
import glob
import numpy as np
from pathlib import Path

from src.detection.detector import VehicleDetector
from src.detection.tracker import Tracker
from src.analysis.zone_analyzer import ZoneAnalyzer


# ── Frame source abstraction ──────────────────────────────────────────────────

class VideoSource:
    """Wraps either a .mp4 file or a folder of JPEGs as a frame iterator."""

    def __init__(self, path):
        self.path = path
        self.is_image_folder = os.path.isdir(path)

        if self.is_image_folder:
            exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            frames = []
            for ext in exts:
                frames.extend(glob.glob(os.path.join(path, ext)))
            self.frames = sorted(frames)
            if not self.frames:
                raise ValueError(f"No images found in {path}")
            # read first frame to get dimensions
            sample = cv2.imread(self.frames[0])
            self.height, self.width = sample.shape[:2]
            self.fps = 25  # UA-DETRAC is 25fps
            self.total = len(self.frames)
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {path}")
            self.fps    = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frames = None

        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_image_folder:
            if self._idx >= len(self.frames):
                raise StopIteration
            frame = cv2.imread(self.frames[self._idx])
            self._idx += 1
            return frame
        else:
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration
            self._idx += 1
            return frame

    def __len__(self):
        return self.total

    def close(self):
        if not self.is_image_folder and hasattr(self, 'cap'):
            self.cap.release()

    @property
    def name(self):
        return Path(self.path).name


def find_sequences(input_dir):
    """
    Find all video sequences in input_dir.
    Returns list of paths — either .mp4/.avi files or subdirectories of images.
    """
    sequences = []

    # video files
    for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
        sequences.extend(glob.glob(os.path.join(input_dir, ext)))

    # image sequence folders (UA-DETRAC style: MVI_XXXXX/)
    for entry in sorted(os.scandir(input_dir)):
        if entry.is_dir():
            images = (glob.glob(os.path.join(entry.path, '*.jpg')) +
                      glob.glob(os.path.join(entry.path, '*.jpeg')) +
                      glob.glob(os.path.join(entry.path, '*.png')))
            if images:
                sequences.append(entry.path)

    return sorted(sequences)


# ── Per-sequence processor ────────────────────────────────────────────────────

def process_sequence(seq_path, detector, output_stats_dir,
                     max_frames=None, video_id=None):
    """
    Run the full pipeline on one sequence.
    Returns path to the per-sequence stats CSV, or None on failure.
    """
    try:
        src = VideoSource(seq_path)
    except ValueError as e:
        print(f"  SKIP: {e}")
        return None

    seq_name = video_id or src.name
    out_csv  = os.path.join(output_stats_dir, f"{seq_name}_stats.csv")

    # skip if already processed
    if os.path.exists(out_csv):
        print(f"  CACHED: {seq_name} (delete to reprocess)")
        return out_csv

    # fresh tracker + zone analyzer per sequence
    # (do NOT reuse across sequences — state would bleed)
    tracker = Tracker(iou_threshold=0.25, max_age=10, min_hits=2)
    zones   = ZoneAnalyzer(src.width, src.height, num_zones=4, fps=src.fps)

    frame_idx  = 0
    limit      = max_frames or len(src)

    for frame in src:
        if frame_idx >= limit:
            break

        detections = detector.detect(frame)
        detections = tracker.update(detections, frame_idx)
        zones.update(detections, tracker, frame_idx)

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    {frame_idx}/{limit} frames")

    src.close()

    if not zones.stats_history and not zones.frame_buffer:
        print(f"  SKIP: no data generated for {seq_name}")
        return None

    # tag every row with the source sequence so we can trace back
    for row in zones.stats_history:
        row['sequence'] = seq_name

    zones.save_stats(out_csv)
    print(f"  → {len(zones.stats_history)}s of stats | "
          f"{zones.num_features} features/timestep")
    return out_csv


# ── CSV combiner ─────────────────────────────────────────────────────────────

def combine_csvs(csv_paths, output_path):
    """
    Concatenate per-sequence CSVs into one big training CSV.

    Each sequence gets its 'second' column reset to be globally unique
    so the LSTM can't accidentally learn sequence boundaries as a pattern.
    We also add a 'sequence' column so you can filter by source later.
    """
    all_rows = []
    fieldnames = None
    global_second = 0

    for path in csv_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        if fieldnames is None:
            fieldnames = list(rows[0].keys())
            # make sure 'sequence' is in there
            if 'sequence' not in fieldnames:
                fieldnames.append('sequence')

        seq_name = rows[0].get('sequence', Path(path).stem)

        for row in rows:
            row['second'] = global_second
            row['sequence'] = seq_name
            all_rows.append(row)
            global_second += 1

    if not all_rows:
        print("No data to combine.")
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nCombined CSV: {len(all_rows)} seconds of data "
          f"from {len(csv_paths)} sequences → {output_path}")
    return output_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main(input_dir, stats_dir, combined_csv, max_frames, limit_seqs):
    os.makedirs(stats_dir, exist_ok=True)

    sequences = find_sequences(input_dir)
    if not sequences:
        print(f"No sequences found in {input_dir}")
        print("Expected: .mp4/.avi files OR subdirectories of JPEG frames (UA-DETRAC style)")
        return

    if limit_seqs:
        sequences = sequences[:limit_seqs]

    print(f"Found {len(sequences)} sequences to process")
    if max_frames:
        print(f"Capped at {max_frames} frames per sequence (~{max_frames//25}s at 25fps)")
    print("=" * 55)

    # one shared detector (expensive to reload per sequence)
    print("Loading YOLOv8 detector...")
    detector = VehicleDetector()

    csv_paths = []
    for i, seq_path in enumerate(sequences):
        seq_name = Path(seq_path).name
        print(f"\n[{i+1}/{len(sequences)}] {seq_name}")
        csv_path = process_sequence(
            seq_path, detector, stats_dir,
            max_frames=max_frames,
            video_id=seq_name
        )
        if csv_path:
            csv_paths.append(csv_path)

    print(f"\n{'='*55}")
    print(f"Processed {len(csv_paths)}/{len(sequences)} sequences successfully.")

    if csv_paths:
        combined = combine_csvs(csv_paths, combined_csv)
        if combined:
            print(f"\nReady to train:")
            print(f"  python -m src.ml.train --input {combined_csv}")
    else:
        print("No data generated. Check your input directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process traffic sequences for LSTM training")
    parser.add_argument('--input',    default='data/raw/',
                        help='Directory containing video files or UA-DETRAC sequence folders')
    parser.add_argument('--stats',    default='data/stats/sequences/',
                        help='Output directory for per-sequence CSVs')
    parser.add_argument('--combined', default='data/stats/combined_training.csv',
                        help='Output path for combined training CSV')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Max frames per sequence (useful for quick testing, e.g. 2500 = 100s)')
    parser.add_argument('--limit',    type=int, default=None,
                        help='Only process first N sequences (for testing)')
    args = parser.parse_args()

    main(args.input, args.stats, args.combined, args.max_frames, args.limit)