!pip install -q ultralytics opencv-python

from ultralytics import YOLO
import cv2
import numpy as np
from google.colab import files
import os

print("Please upload your traffic video file: ")
uploaded = files.upload()

VIDEO_PATH = next(iter(uploaded.keys()))
print("Using video:", VIDEO_PATH)

MANUAL_HEAVY_COUNT = 5

model = YOLO("yolov8n.pt")

HEAVY_CLASSES = {"bus", "truck"}

CONF_THRESH = 0.4
ASSOC_DIST_THRESHOLD = 60
MAX_MISSED_FRAMES = 10

class Track:
    def __init__(self, track_id, centroid, bbox, class_name):
        self.id = track_id
        self.centroid = centroid
        self.prev_centroid = centroid
        self.bbox = bbox
        self.class_name = class_name
        self.counted = False
        self.missed = 0

tracks = {}
next_track_id = 0

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps == 0:
    fps = 25.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = "output_heavy_vehicles.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

line_y = height // 2

automatic_heavy_count = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    results = model(frame, conf=CONF_THRESH, verbose=False)[0]

    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]

            if class_name not in HEAVY_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "centroid": (cx, cy),
                "class_name": class_name,
                "conf": conf,
            })

    for t in tracks.values():
        t.missed += 1

    for det in detections:
        cx, cy = det["centroid"]

        best_track_id = None
        best_dist = ASSOC_DIST_THRESHOLD + 1

        for track_id, track in tracks.items():
            dist = np.hypot(cx - track.centroid[0], cy - track.centroid[1])
            if dist < best_dist and dist <= ASSOC_DIST_THRESHOLD:
                best_dist = dist
                best_track_id = track_id

        if best_track_id is not None:
            track = tracks[best_track_id]
            track.prev_centroid = track.centroid
            track.centroid = det["centroid"]
            track.bbox = det["bbox"]
            track.class_name = det["class_name"]
            track.missed = 0
        else:
            track = Track(next_track_id, det["centroid"], det["bbox"], det["class_name"])
            tracks[next_track_id] = track
            next_track_id += 1

    tracks = {
        tid: t for tid, t in tracks.items()
        if t.missed <= MAX_MISSED_FRAMES
    }

    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

    for track in tracks.values():
        x1, y1, x2, y2 = track.bbox
        cx, cy = track.centroid

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{track.class_name} ID {track.id}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        if not track.counted:
            y_prev = track.prev_centroid[1]
            y_curr = track.centroid[1]

            crossed_down = y_prev < line_y <= y_curr
            crossed_up = y_prev > line_y >= y_curr

            if crossed_down or crossed_up:
                automatic_heavy_count += 1
                track.counted = True

    cv2.putText(frame,
                f"Heavy vehicles counted: {automatic_heavy_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2)

    out.write(frame)

cap.release()
out.release()

print("=======================================")
print("Manual heavy vehicle count (ground truth):", MANUAL_HEAVY_COUNT)
print("Automatic heavy vehicle count:", automatic_heavy_count)

if MANUAL_HEAVY_COUNT > 0:
    accuracy = 100.0 * (1.0 - abs(automatic_heavy_count - MANUAL_HEAVY_COUNT) / MANUAL_HEAVY_COUNT)
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("Accuracy: N/A (manual count is 0)")

print("Annotated output video saved to:", output_path)

files.download(output_path)
