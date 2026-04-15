# Heavy Vehicle Detection and Counting using YOLOv8 (Computer Vision)

## Overview
This project implements a computer vision system to detect, track, and count heavy vehicles (buses and trucks) from a traffic video using YOLOv8.

The system processes a video, tracks moving vehicles across frames, counts them when they cross a defined line, and evaluates accuracy against a manually counted ground truth.

---

## Problem Statement
Record a real-world traffic video (20–30 seconds) and:
- Detect heavy vehicles (trucks, buses)
- Track their movement
- Count them automatically
- Compare with manual count and compute accuracy

---

## Why I Built This
I built this project to:
- Apply object detection using YOLOv8
- Understand object tracking across frames
- Implement real-time counting logic
- Evaluate model performance with accuracy metrics

---

## Methodology

1. Video Input:
   - User provides a traffic video

2. Object Detection:
   - YOLOv8 model detects vehicles in each frame

3. Filtering:
   - Only heavy vehicles (bus, truck) are considered

4. Object Tracking:
   - Tracks objects using centroid-based tracking
   - Assigns unique IDs to vehicles

5. Counting Logic:
   - A horizontal line is defined
   - Vehicles are counted when they cross the line

6. Accuracy Evaluation:
   - Compare automatic count with manual count
   - Compute percentage accuracy

---

## Key Concepts Used
- YOLOv8 object detection
- Object tracking (centroid-based)
- Video processing with OpenCV
- Real-time counting logic
- Performance evaluation (accuracy %)

---

## Tech Stack
- Python
- OpenCV
- NumPy
- Ultralytics YOLOv8

---

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run the script / notebook:
python main.py

(If using Jupyter Notebook, run all cells step by step)

3. Upload or provide your own traffic video when prompted

4. Set manual vehicle count (ground truth) in the code:
MANUAL_HEAVY_COUNT = your_count

5. Run the full pipeline to:
- Detect and track vehicles
- Count heavy vehicles
- Generate annotated output video

---

## Input
- A traffic video (20–30 seconds) recorded by the user
  
---

## Output
- Annotated video with:
  - Bounding boxes
  - Vehicle IDs
  - Count overlay
- Final statistics:
  - Automatic vehicle count
  - Manual count
  - Accuracy percentage

Output file:
output_heavy_vehicles.mp4
