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

# Process each frame of the video
while cap.isOpened():

    # Capture a single frame from the video
    success, frame = cap.read()
    if success:
        # Apply the YOLOv11 model to track objects in the frame
        results = model.track(frame, persist=True, save=False)
        boxes = results[0].boxes.xywh.cpu() 
        track_ids = results[0].boxes.id.int().cpu().tolist() 
        annotated_frame = results[0].plot() 
        
        # Loop through the detected objects and their respective tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id] 
            track.append((float(x), float(y)))  
          
            
        cv2.namedWindow('YOLOv11 Tracking', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("YOLOv11 Tracking", annotated_frame)
        window = cv2.resizeWindow('YOLOv11 Tracking', 1240, 700)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
