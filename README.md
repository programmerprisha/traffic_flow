Traffic Flow Analysis Using Computer Vision

A computer vision-based traffic monitoring system that detects and tracks vehicles in traffic videos, estimates vehicle speeds, analyzes traffic conditions, and prepares traffic statistics for machine learning.

Overview

The Traffic Flow Analysis project uses computer vision and machine learning techniques to analyze road traffic from video footage.

The system detects vehicles in video frames, tracks their movement across consecutive frames, estimates their speeds, and analyzes traffic conditions across different regions of the video. It can also process multiple video sequences and prepare datasets for training an LSTM-based model.

Features

Vehicle Detection: Detects cars, motorcycles, buses, and trucks using YOLOv8.

Vehicle Tracking: Assigns tracking IDs to detected vehicles and follows their movement across frames.

Speed Estimation: Estimates vehicle speeds using changes in position between frames.

Traffic Zone Analysis: Divides the video frame into zones and calculates traffic statistics for each zone.

Congestion Analysis: Estimates traffic congestion based on vehicle count, speed, density, and occupancy.

Video Annotation: Draws bounding boxes, tracking IDs, and speed labels on traffic videos.

Batch Processing: Processes multiple traffic videos or image sequences.

Dataset Generation: Exports traffic statistics to CSV files for further analysis.

LSTM Model Training: Provides a training pipeline for learning patterns from traffic data.

Technologies Used

Python — Core programming language

OpenCV — Video processing and annotation

YOLOv8 (Ultralytics) — Vehicle detection

NumPy — Numerical calculations

SciPy — Optimization algorithms for tracking

PyTorch — Deep learning and LSTM training

Pandas — Traffic data processing and CSV handling

Project Structure

traffic_flow/
├── data/
│   ├── raw/                  # Input traffic videos
│   ├── annotated/            # Processed output videos
│   └── stats/                # Traffic statistics and CSV datasets
├── src/
│   ├── detection/
│   │   ├── detector.py      # Vehicle detection
│   │   └── tracker.py       # Vehicle tracking
│   ├── analysis/
│   │   └── zone_analyzer.py # Traffic zone analysis
│   └── ml/
│       └── train.py          # LSTM model training
├── tests/                    # Project tests
├── main.py                   # Main video-processing script
├── batch_process.py          # Batch video processing
├── requirements.txt          # Python dependencies
└── README.md

Note: This structure summarizes the main project components; individual filenames and directories should be checked against the current repository.

Installation

1. Clone the repository

git clone https://github.com/programmerprisha/traffic_flow.git
cd traffic_flow

2. Create a virtual environment

python -m venv .venv

Activate the environment.

Windows:

.venv\Scripts\activate

Linux/macOS:

source .venv/bin/activate

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

PyTorch installation requirements may vary depending on your operating system and hardware. Refer to the official PyTorch installation guide if needed.

Usage

Run vehicle detection and tracking

Place a traffic video in the expected input location:

data/raw/kaggle.mp4

Run the main program:

python main.py

The script is configured to save an annotated video at:

data/annotated/annotated_kaggle.mp4

The output is intended to display detected vehicles, tracking IDs, and estimated speeds.

Note: The input path, output path, and video codec may need to be adjusted for your environment.

Process multiple videos

Place your input videos in data/raw/ and run:

python batch_process.py --input data/raw/

The batch processor generates traffic statistics for individual sequences and combines them into a CSV dataset for further analysis or model training.

Train the LSTM model

After generating the combined traffic dataset, run:

python -m src.ml.train --input data/stats/combined_training.csv

The training script uses historical traffic features to train an LSTM model.

Ensure that the CSV file contains the columns and target values expected by the training pipeline before starting training.

How It Works

The system follows these main steps:

Video Input: Reads a traffic video frame by frame using OpenCV.

Vehicle Detection: Uses YOLOv8 to identify vehicles in each frame.

Vehicle Tracking: Associates detections across frames and assigns tracking IDs.

Speed Estimation: Estimates vehicle movement and speed from changes in position.

Traffic Analysis: Calculates vehicle counts, speed statistics, occupancy, density, and congestion scores for different zones.

Data Export: Saves traffic statistics in CSV format.

Machine Learning: Uses the generated data to train an LSTM model for learning temporal traffic patterns.

Traffic Analysis Metrics

The project analyzes several traffic-related measurements:

Metric

Description

Vehicle Count

Number of tracked vehicles observed in a zone

Average Speed

Average estimated speed of vehicles

Vehicle Density

Vehicle count relative to the zone's dimensions

Occupancy

Proportion of image area covered by vehicle bounding boxes

Vehicle Composition

Distribution of detected vehicle types

Traffic Congestion

A score derived from selected traffic features

Zone Transitions

Movement of tracked vehicles between zones
