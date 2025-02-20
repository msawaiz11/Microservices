import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os
import cv2
import csv
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from ultralytics import YOLO
import pandas as pd

# Import class mappings
from All_App.utils.class_mapping import yolo_classes, numberplate_classes

# Load YOLO models
# model = YOLO(r"weights\yolo11n.pt")
# numberplate_model = YOLO(r"weights\np.pt")


from pathlib import Path
from ultralytics import YOLO



yolo_model = YOLO(r"E:\P_M_services\All_Project\weights\yolo11n.pt", verbose=False).to('cuda')
numberplate_model = YOLO(r"E:\P_M_services\All_Project\weights\np.pt", verbose=False).to('cuda')



# Initialize video capture


# Object tracking storage
tracked_objects = {}  # Stores final saved ROIs
object_buffers = {}  # Stores recent frames of detected objects
object_max_areas = {}  # Stores maximum area seen for each tracked object
object_entry_frames = {}  # Stores the frame number when object first detected

# Parameters for stabilization
# BUFFER_SIZE = 10  # Frames to track size consistency
# STABLE_THRESHOLD = 3  # Frames object size must be stable before saving
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to accept detection
# IOU_THRESHOLD = 0.5  # IoU threshold to check overlap
# AREA_GROWTH_THRESHOLD = 0.2  # Minimum area growth to trigger new save
# MIN_FRAMES_BETWEEN_SAVES = fps * 2  # Minimum frames between saves of same object



def compute_iou(box1, box2):
    """Computes IoU between two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_timestamp(frame_count, fps, start_time):
    """Calculate actual timestamp and date based on frame count and start time."""
    seconds_elapsed = frame_count / fps
    current_time = start_time + timedelta(seconds=seconds_elapsed)
    return current_time.strftime("%Y-%m-%d"), current_time.strftime("%H:%M:%S.%f")[:-4]

def load_model(model_path):
    """Load the YOLO model."""
    return YOLO(model_path)

def process_frame(frame, yolo_model):
    """Process a single frame for object detection."""
    results = yolo_model.track(frame, persist=True, verbose=False)
    # return results
    return results if results else []




# def save_detection_info(csv_file, detection_info):
#     """Save detection information to CSV."""
#     os.makedirs(os.path.dirname(csv_file), exist_ok=True)
#     with open(csv_file, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         # writer.writerow(["Date", "Timestamp", "Class", "Track_ID", "Confidence", "IoU", "Area_Growth_Rate", "X1", "Y1", "X2", "Y2", "ROI_Path"])
#         writer.writerow(detection_info)


def save_detection_info(csv_file, detection_info):
    """Save detection information to CSV."""
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Check if file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file does not exist
        if not file_exists:
            writer.writerow(["Date", "Timestamp", "Class", "Track_ID", "Confidence", "IoU", "Area_Growth_Rate", "X1", "Y1", "X2", "Y2", "ROI_Path"])
        
        writer.writerow(detection_info)  # Append detection data


###  main function ###


'''
def process_video(video_path, selected_classes):
    # Initialize results list
    results = []

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        date, timestamp = get_timestamp(frame_count, fps, video_start_time)

        # Process the frame for YOLOv11 model
        results_yolo = process_frame(frame, model)

        if results_yolo[0].boxes is not None:
            for box in results_yolo[0].boxes:
                # Extract bounding box and class info
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id) if box.id is not None else -1
                confidence = float(box.conf[0])
                class_name = model.names[int(box.cls[0])]

                # Check if the detected class is in the user's selection
                if confidence >= CONFIDENCE_THRESHOLD and (int(box.cls[0]) + 1) in selected_classes:
                    roi_path = os.path.join(output_folder, f"roi_{track_id}.jpg")
                    cv2.imwrite(roi_path, frame[y1:y2, x1:x2])  # Save ROI

                    # Append detection to results
                    results.append({
                        "date": date,
                        "time": timestamp,
                        "class": class_name,
                        "probability": confidence,
                        "roi": roi_path
                    })

    cap.release()
    return results

'''