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
            
            # Limit the track history to the last 30 positions
            if len(track) > 30:
                track.pop(0)
            
            # Prepare the points for drawing a polyline of the track path
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

            # Draw the polyline representing the object's movement path
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Placeholder array for tracking purposes (unused in this code)
            empt = np.array([[0, 0]])

            # Calculate the difference between the first and last points in the track 
            difference = np.array(points[0] - points[-1])
            
            print(difference)
            
        cv2.namedWindow('YOLOv11 Tracking', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("YOLOv11 Tracking", annotated_frame)
        window = cv2.resizeWindow('YOLOv11 Tracking', 1240, 700)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
