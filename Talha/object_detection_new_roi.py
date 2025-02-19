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
from class_mapping import yolo_classes, numberplate_classes

# Load YOLO models
# model = YOLO(r"weights\yolo11n.pt")
# numberplate_model = YOLO(r"weights\np.pt")

model = YOLO(r"All_Project\weights\yolo11n.pt").to('cuda')
numberplate_model = YOLO(r"All_Project\weights\np.pt").to('cuda')

# Input video path
video_path = r"C:\Users\Msawaiz10\Desktop\final.mkv"

# Extract video name without extension for folder naming
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Create a unique output folder name
output_folder = os.path.join(os.getcwd(), f"{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Get video start time (current time for this example)
video_start_time = datetime.now()

# CSV file to store detections
csv_file = os.path.join(output_folder, f"{video_name}_detections.csv")

# Open CSV file and write headers
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Date", "Timestamp", "Class", "Track_ID", "Confidence", "IoU", "Area_Growth_Rate", "X1", "Y1", "X2", "Y2", "ROI_Path"])

# Initialize video capture
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = 0

# Object tracking storage
tracked_objects = {}  # Stores final saved ROIs
object_buffers = {}  # Stores recent frames of detected objects
object_max_areas = {}  # Stores maximum area seen for each tracked object
object_entry_frames = {}  # Stores the frame number when object first detected

# Parameters for stabilization
BUFFER_SIZE = 10  # Frames to track size consistency
STABLE_THRESHOLD = 3  # Frames object size must be stable before saving
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to accept detection
IOU_THRESHOLD = 0.5  # IoU threshold to check overlap
AREA_GROWTH_THRESHOLD = 0.2  # Minimum area growth to trigger new save
MIN_FRAMES_BETWEEN_SAVES = fps * 2  # Minimum frames between saves of same object

# def get_user_selection():
#     """Prompt user to select classes for detection."""
#     print("Select classes to detect from YOLOv11 model:")
#     for key, value in yolo_classes.items():
#         print(f"{key}) {value}")
#     selected_yolo_classes = input("Enter the numbers of the classes you want to detect (comma-separated): ")
#     selected_yolo_classes = [int(x) for x in selected_yolo_classes.split(",") if x.isdigit()]

#     print("\nSelect classes to detect from Number Plate model:")
#     for key, value in numberplate_classes.items():
#         print(f"{key}) {value}")
#     selected_numberplate_classes = input("Enter the numbers of the classes you want to detect (comma-separated): ")
#     selected_numberplate_classes = [int(x) for x in selected_numberplate_classes.split(",") if x.isdigit()]

#     return selected_yolo_classes, selected_numberplate_classes

# # Get user selections
# selected_yolo_classes, selected_numberplate_classes = get_user_selection()

selected_yolo_classes = 1
selected_numberplate_classes = 1


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

def process_frame(frame, model):
    """Process a single frame for object detection."""
    results = model.track(frame, persist=True)
    return results

def save_detection_info(csv_file, detection_info):
    """Save detection information to CSV."""
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(detection_info)




###  main function ###

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

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    date, timestamp = get_timestamp(frame_count, fps, video_start_time)

    # Process the frame for YOLOv11 model
    results = process_frame(frame, model)

    if results[0].boxes is not None:
        for box in results[0].boxes:
            # Extract bounding box and class info
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id) if box.id is not None else -1
            confidence = float(box.conf[0])
            class_name = model.names[int(box.cls[0])]

            # Check if the detected class is in the user's selection
            if confidence >= CONFIDENCE_THRESHOLD and (int(box.cls[0]) + 1) in selected_yolo_classes:
                # Calculate current bbox area
                current_area = (x2 - x1) * (y2 - y1)

                # Initialize tracking for new objects
                if track_id not in object_buffers:
                    object_buffers[track_id] = deque(maxlen=BUFFER_SIZE)
                    object_max_areas[track_id] = current_area
                    object_entry_frames[track_id] = frame_count

                # Update maximum area if current area is larger
                if current_area > object_max_areas[track_id]:
                    object_max_areas[track_id] = current_area

                # Calculate area growth rate compared to initial detection
                initial_area = object_buffers[track_id][0][4] if object_buffers[track_id] else current_area
                area_growth_rate = (current_area - initial_area) / initial_area if initial_area > 0 else 0

                # Store current ROI in buffer
                object_buffers[track_id].append((x1, y1, x2, y2, current_area, frame_count, confidence))

                # Check if we should save this detection
                if len(object_buffers[track_id]) == BUFFER_SIZE:
                    recent_sizes = [b[4] for b in list(object_buffers[track_id])[-STABLE_THRESHOLD:]]
                    size_variation = max(recent_sizes) - min(recent_sizes)
                    avg_size = sum(recent_sizes) / len(recent_sizes)

                    # Conditions for saving:
                    if size_variation / avg_size < 0.05:
                        save_detection = False
                        current_iou = 0.0

                        if track_id in tracked_objects:
                            prev_roi = tracked_objects[track_id]
                            current_iou = compute_iou(
                                (x1, y1, x2, y2),
                                (prev_roi[0], prev_roi[1], prev_roi[2], prev_roi[3])
                            )
                            
                            # Check if enough frames have passed and area has grown
                            frames_since_last_save = frame_count - prev_roi[5]
                            if frames_since_last_save >= MIN_FRAMES_BETWEEN_SAVES:
                                if area_growth_rate > AREA_GROWTH_THRESHOLD:
                                    save_detection = True
                        else:
                            # First detection of this object
                            save_detection = True

                        if save_detection:
                            # Save the current ROI
                            tracked_objects[track_id] = (x1, y1, x2, y2, current_area, frame_count, confidence)
                            roi_path = os.path.join(output_folder, 
                                                  f"{track_id}_{class_name}_{frame_count}.jpg")
                            cv2.imwrite(roi_path, frame[y1:y2, x1:x2])

                            # Save detection info to CSV
                            save_detection_info(csv_file, [
                                date, timestamp, class_name, track_id, confidence,
                                current_iou, area_growth_rate, x1, y1, x2, y2, roi_path
                            ])

                # Draw bounding box and label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 
                           f"{class_name} {track_id} ({confidence:.2f})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Process the frame for Number Plate model
    results_np = process_frame(frame, numberplate_model)

    if results_np[0].boxes is not None:
        for box in results_np[0].boxes:
            # Extract bounding box and class info
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id) if box.id is not None else -1
            confidence = float(box.conf[0])
            class_name = numberplate_model.names[int(box.cls[0])]

            # Check if the detected class is in the user's selection
            if confidence >= CONFIDENCE_THRESHOLD and (int(box.cls[0]) + 1) in selected_numberplate_classes:
                # Calculate current bbox area
                current_area = (x2 - x1) * (y2 - y1)

                # Initialize tracking for new objects
                if track_id not in object_buffers:
                    object_buffers[track_id] = deque(maxlen=BUFFER_SIZE)
                    object_max_areas[track_id] = current_area
                    object_entry_frames[track_id] = frame_count

                # Update maximum area if current area is larger
                if current_area > object_max_areas[track_id]:
                    object_max_areas[track_id] = current_area

                # Calculate area growth rate compared to initial detection
                initial_area = object_buffers[track_id][0][4] if object_buffers[track_id] else current_area
                area_growth_rate = (current_area - initial_area) / initial_area if initial_area > 0 else 0

                # Store current ROI in buffer
                object_buffers[track_id].append((x1, y1, x2, y2, current_area, frame_count, confidence))

                # Check if we should save this detection
                if len(object_buffers[track_id]) == BUFFER_SIZE:
                    recent_sizes = [b[4] for b in list(object_buffers[track_id])[-STABLE_THRESHOLD:]]
                    size_variation = max(recent_sizes) - min(recent_sizes)
                    avg_size = sum(recent_sizes) / len(recent_sizes)

                    # Conditions for saving:
                    if size_variation / avg_size < 0.05:
                        save_detection = False
                        current_iou = 0.0

                        if track_id in tracked_objects:
                            prev_roi = tracked_objects[track_id]
                            current_iou = compute_iou(
                                (x1, y1, x2, y2),
                                (prev_roi[0], prev_roi[1], prev_roi[2], prev_roi[3])
                            )
                            
                            # Check if enough frames have passed and area has grown
                            frames_since_last_save = frame_count - prev_roi[5]
                            if frames_since_last_save >= MIN_FRAMES_BETWEEN_SAVES:
                                if area_growth_rate > AREA_GROWTH_THRESHOLD:
                                    save_detection = True
                        else:
                            # First detection of this object
                            save_detection = True

                        if save_detection:
                            # Save the current ROI
                            tracked_objects[track_id] = (x1, y1, x2, y2, current_area, frame_count, confidence)
                            roi_path = os.path.join(output_folder, 
                                                  f"{track_id}_{class_name}_{frame_count}.jpg")
                            cv2.imwrite(roi_path, frame[y1:y2, x1:x2])

                            # Save detection info to CSV
                            save_detection_info(csv_file, [
                                date, timestamp, class_name, track_id, confidence,
                                current_iou, area_growth_rate, x1, y1, x2, y2, roi_path
                            ])

                # Draw bounding box and label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Different color for number plates
                cv2.putText(frame, 
                           f"{class_name} {track_id} ({confidence:.2f})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
