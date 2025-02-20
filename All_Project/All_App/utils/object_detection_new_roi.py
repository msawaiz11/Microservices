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
from typing import Dict, List, Tuple, Optional



yolo_model = YOLO(r"E:\P_M_services\All_Project\weights\yolo11n.pt", verbose=False).to('cuda')
numberplate_model = YOLO(r"E:\P_M_services\All_Project\weights\np.pt", verbose=False).to('cuda')



# Initialize video capture


# Object tracking storage
tracked_objects = {}  # Stores final saved ROIs
object_buffers = {}  # Stores recent frames of detected objects
object_max_areas = {}  # Stores maximum area seen for each tracked object
object_entry_frames = {}  # Stores the frame number when object first detected


CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to accept detection





config = {
        'confidence_threshold': 0.7,  # Adjust confidence threshold as needed
        # Optionally override other parameters
        'buffer_size': 10,
        'stable_threshold': 3,
        'iou_threshold': 0.5,
        'area_growth_threshold': 0.2,
        'min_frames_between_saves': None
    }




def setup_output(video_path: str, base_output_dir: Optional[str] = None) -> Tuple[str, str, str]:
    """Setup output directory and CSV file"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{video_name}_{timestamp}"
    
    if base_output_dir:
        output_folder = os.path.join(base_output_dir, folder_name)
    else:
        output_folder = os.path.join(os.getcwd(), folder_name)
    
    os.makedirs(output_folder, exist_ok=True)
    csv_file = os.path.join(output_folder, f"{video_name}_detections.csv")
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Timestamp", "Class", "Track_ID", "Confidence", 
                        "IoU", "Area_Growth_Rate", "X1", "Y1", "X2", "Y2", "ROI_Path"])
    
    return output_folder, csv_file, folder_name



def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two bounding boxes"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def load_model(model_path):
    """Load the YOLO model."""
    return YOLO(model_path)






def process_detection(frame, box, model, class_name, frame_count, date, timestamp,
                     tracked_objects, object_buffers, object_max_areas,
                     config, output_folder, csv_file, video_path):
    """Process a single detection"""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    track_id = int(box.id) if box.id is not None else -1
    confidence = float(box.conf[0])
    current_area = (x2 - x1) * (y2 - y1)
    current_iou = 0.0



    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    config['min_frames_between_saves'] = fps * 2


    if track_id not in object_buffers:
        object_buffers[track_id] = deque(maxlen=config['buffer_size'])
        object_max_areas[track_id] = current_area

    if current_area > object_max_areas[track_id]:
        object_max_areas[track_id] = current_area

    initial_area = object_buffers[track_id][0][4] if object_buffers[track_id] else current_area
    area_growth_rate = (current_area - initial_area) / initial_area if initial_area > 0 else 0

    object_buffers[track_id].append((x1, y1, x2, y2, current_area, frame_count, confidence))

    if len(object_buffers[track_id]) == config['buffer_size']:
        recent_sizes = [b[4] for b in list(object_buffers[track_id])[-config['stable_threshold']:]]
        size_variation = max(recent_sizes) - min(recent_sizes)
        avg_size = sum(recent_sizes) / len(recent_sizes)

        if size_variation / avg_size < 0.05:
            should_save = False

            if track_id in tracked_objects:
                prev_roi = tracked_objects[track_id]
                current_iou = compute_iou(
                    (x1, y1, x2, y2),
                    (prev_roi[0], prev_roi[1], prev_roi[2], prev_roi[3])
                )
                
                frames_since_last_save = frame_count - prev_roi[5]
                if frames_since_last_save >= config['min_frames_between_saves']:
                    if area_growth_rate > config['area_growth_threshold']:
                        should_save = True
            else:
                should_save = True

            if should_save:
                tracked_objects[track_id] = (x1, y1, x2, y2, current_area, frame_count, confidence)
                roi_path = os.path.join(output_folder, f"{track_id}_{class_name}_{frame_count}.jpg")
                cv2.imwrite(roi_path, frame[y1:y2, x1:x2])

                detection_info = [
                    date, timestamp, class_name, track_id, confidence,
                    current_iou, area_growth_rate, x1, y1, x2, y2, roi_path
                ]
                save_detection(detection_info, csv_file)

                return detection_info

    return None



def process_frame(frame, frame_count, date, timestamp, yolo_model, numberplate_model,
                 tracked_objects, object_buffers, object_max_areas, config,
                 output_folder, csv_file,selected_classes,selected_numberplate_classes,video_path):
    """Process a single frame"""
    detections = []
    
    # Process YOLO model only if selected_yolo_classes is not None and has values
    if yolo_model and selected_classes:
        # print("inside the yolo")
        results = yolo_model.track(frame, persist=True, verbose=False)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
                       
            for box in results[0].boxes:
                if (float(box.conf[0]) >= config['confidence_threshold'] and 
                    (int(box.cls[0]) + 1) in selected_classes):
                    class_name = yolo_model.names[int(box.cls[0])]



                    detection = process_detection(
                        frame, box, yolo_model, class_name, frame_count, date, timestamp,
                        tracked_objects, object_buffers, object_max_areas, config,
                        output_folder, csv_file, video_path
                    )

                    if detection:
                        detections.append(detection)
                    




    # Process numberplate model only if selected_numberplate_classes is not None and has values
    if numberplate_model and selected_numberplate_classes:
        # print("inside the number plate model")
        results = numberplate_model.track(frame, persist=True,verbose=False)
        # print("results", results)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if (float(box.conf[0]) >= config['confidence_threshold'] and 
                    (int(box.cls[0]) + 1) in selected_numberplate_classes):
                    class_name = numberplate_model.names[int(box.cls[0])]
                    detection = process_detection(
                        frame, box, numberplate_model, class_name, frame_count, date, timestamp,
                        tracked_objects, object_buffers, object_max_areas, config,
                        output_folder, csv_file, video_path
                    )
                    if detection:
                        detections.append(detection)

    return detections




def get_timestamp(frame_count: int, fps: int, start_time: datetime) -> Tuple[str, str]:
    """Calculate timestamp based on frame count"""
    seconds_elapsed = frame_count / fps
    current_time = start_time + timedelta(seconds=seconds_elapsed)
    return current_time.strftime("%Y-%m-%d"), current_time.strftime("%H:%M:%S.%f")[:-4]



def save_detection(detection_info: List, csv_file: str):
    """Save detection information to CSV"""
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(detection_info)
###  main function ###