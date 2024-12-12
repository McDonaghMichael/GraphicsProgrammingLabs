import cv2
import numpy as np
from collections import defaultdict

from ultralytics import YOLO

# Load the YOLOv11 model
model = YOLO("yolo11n.pt")
track_history = defaultdict(lambda: [])  

# Path to the input video
video_path = "runningvideo.mp4"
cap = cv2.VideoCapture(video_path)  # Open the video for processing

