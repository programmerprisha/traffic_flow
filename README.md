# Traffic Flow Analysis Using Computer Vision

A computer vision-based traffic monitoring system that detects and tracks vehicles in traffic videos, estimates vehicle speeds, analyzes traffic conditions, and generates traffic statistics for machine learning.

## Overview

The Traffic Flow Analysis project uses computer vision and machine learning techniques to analyze road traffic from video footage.

The system detects vehicles in video frames, tracks their movement across consecutive frames, estimates their speeds, and analyzes traffic conditions across different regions of the video. It can also process multiple video sequences and prepare datasets for training an LSTM-based model.

## Features

- **Vehicle Detection:** Detects cars, motorcycles, buses, and trucks using YOLOv8.
- **Vehicle Tracking:** Assigns tracking IDs to detected vehicles and follows their movement across frames.
- **Speed Estimation:** Estimates vehicle speeds using changes in position between frames.
- **Traffic Zone Analysis:** Divides video frames into zones and calculates traffic statistics for each zone.
- **Congestion Analysis:** Estimates traffic congestion based on vehicle count, speed, density, and occupancy.
- **Video Annotation:** Draws bounding boxes, tracking IDs, and speed labels on traffic videos.
- **Batch Processing:** Processes multiple traffic videos or image sequences.
- **Dataset Generation:** Exports traffic statistics to CSV files for further analysis.
- **LSTM Model Training:** Provides a training pipeline for learning temporal patterns from traffic data.

## Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy
- SciPy
- PyTorch
- Pandas

## Project Structure

```text
traffic_flow/
├── data/
│   ├── raw/                  # Input traffic videos
│   ├── annotated/            # Processed output videos
│   └── stats/                # Traffic statistics and CSV datasets
├── src/
│   ├── detection/
│   │   ├── detector.py       # Vehicle detection
│   │   └── tracker.py        # Vehicle tracking
│   ├── analysis/
│   │   └── zone_analyzer.py  # Traffic zone analysis
│   └── ml/
│       └── train.py          # LSTM model training
├── tests/                    # Project tests
├── main.py                   # Main video-processing script
├── batch_process.py          # Batch video processing
├── requirements.txt          # Python dependencies
└── README.md
