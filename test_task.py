

import cv2
import os
from datetime import datetime
from collections import deque
from celery import shared_task

# Define missing variables (Ensure they are set correctly in your real implementation)
object_buffers = {}
object_max_areas = {}
object_entry_frames = {}

@shared_task
def Object_Detection_Celery(video_path, check_box_names, selected_classes):
    print("Selected classes:", selected_classes)
    print("Checkbox names:", check_box_names)


    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create a unique output folder name
    output_folder = os.path.join(os.getcwd(), f"{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Get video start time (current time for this example)
    video_start_time = datetime.now()

    # CSV file to store detections
    csv_file = os.path.join(output_folder, f"{video_name}_detections.csv")






    import ast
    # Ensure check_box_names is always a list
    if isinstance(check_box_names, str):
        try:
            check_box_names = ast.literal_eval(check_box_names)  # Convert string representation of a list
        except (ValueError, SyntaxError):
            check_box_names = [check_box_names]  # If it's a single string, treat it as a list

    print(f"Checkbox names received: {check_box_names}")  

    # Ensure selected_classes is always a list of integers
    if isinstance(selected_classes, str):
        try:
            selected_classes = ast.literal_eval(selected_classes)  # Convert string representation of list
            if isinstance(selected_classes, int):  
                selected_classes = [selected_classes]  # Convert single integer to a list
            elif isinstance(selected_classes, list):
                selected_classes = [int(cls) for cls in selected_classes]  # Convert all elements to integers
        except (ValueError, SyntaxError):
            selected_classes = []  # Fallback to empty list

    print(f"Processed selected_classes: {selected_classes}, Type: {type(selected_classes)}")  

    # Logic to set handle_other_objects and handle_number_plate correctly
    handle_other_objects = any(name != "Number_Plate" for name in check_box_names) if check_box_names else False
    handle_number_plate = "Number_Plate" in check_box_names

    print(f"handle_other_objects: {handle_other_objects}, handle_number_plate: {handle_number_plate}")





    # Corrected condition
    handle_other_objects = any(name != "Number_Plate" for name in check_box_names) if check_box_names else False
    handle_number_plate = "Number_Plate" in check_box_names

    print(f"Final handle_other_objects: {handle_other_objects}")  # Should be False if only "Number_Plate" exists
    print(f"Final handle_number_plate: {handle_number_plate}")    # Should be True if "Number_Plate" is present



    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return {"error": "Invalid video file"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    MIN_FRAMES_BETWEEN_SAVES = fps * 2
    video_start_time = datetime.now()
    frame_count = 0

    results = []
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(os.getcwd(), f"{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_folder, exist_ok=True)

    if handle_other_objects:
        # print("Processing other objects...")
        print("selected classes", selected_classes)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            date, timestamp = get_timestamp(frame_count, fps, video_start_time)

            results_yolo = process_frame(frame, yolo_model)
            
            if results_yolo and hasattr(results_yolo[0], "boxes"):
                for box in results_yolo[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id) if box.id is not None else -1
                    confidence = float(box.conf[0])
                    class_index = int(box.cls[0]) + 1  

                    if 0 <= class_index - 1 < len(yolo_model.names):
                        class_name = yolo_model.names[class_index - 1]
                    else:
                        continue  # Skip if index is out of range
                    print("type of sleected class", type(selected_classes))
                    if confidence >= CONFIDENCE_THRESHOLD and class_index in selected_classes:
                        roi_path = os.path.join(output_folder, f"roi_{track_id}.jpg")
                        cv2.imwrite(roi_path, frame[y1:y2, x1:x2])  

                        results.append({
                            "date": date,
                            "time": timestamp,
                            "class": class_name,
                            "probability": confidence,
                            "roi": roi_path
                        })

        # print("Detection results:", results)

    if handle_number_plate:
        print("Handling number plate detection...")
        # selected_numberplate_classes = "1"
        selected_numberplate_classes = [1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
        frame_count = 0  # Reset frame count

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame") 
                break

            frame_count += 1
            date, timestamp = get_timestamp(frame_count, fps, video_start_time)

            # print("Processing frame for number plate detection...")  # Debugging output
            
            results_yolo = process_frame(frame, yolo_model)
            # print(f"Results received: {results_yolo}")  # Debugging output
            # exit()
            if results_yolo and hasattr(results_yolo[0], "boxes"):
                # print("inside boxes")
                for box in results_yolo[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id) if box.id is not None else -1
                    confidence = float(box.conf[0])
                    class_name = yolo_model.names[int(box.cls[0])]

                    # print(f"Detected: {class_name} (Confidence: {confidence})") 

                    if confidence >= CONFIDENCE_THRESHOLD and (int(box.cls[0]) + 1) in selected_classes:
                        current_area = (x2 - x1) * (y2 - y1)

                        if track_id not in object_buffers:
                            object_buffers[track_id] = deque(maxlen=BUFFER_SIZE)
                            object_max_areas[track_id] = current_area
                            object_entry_frames[track_id] = frame_count

                        # Update maximum area if current area is larger
                        if current_area > object_max_areas[track_id]:
                            object_max_areas[track_id] = current_area

                        initial_area = object_buffers[track_id][0][4] if object_buffers[track_id] else current_area
                        area_growth_rate = (current_area - initial_area) / initial_area if initial_area > 0 else 0

                        # Store current ROI in buffer
                        object_buffers[track_id].append((x1, y1, x2, y2, current_area, frame_count, confidence))

                        if len(object_buffers[track_id]) == BUFFER_SIZE:
                                recent_sizes = [b[4] for b in list(object_buffers[track_id])[-STABLE_THRESHOLD:]]
                                size_variation = max(recent_sizes) - min(recent_sizes)
                                avg_size = sum(recent_sizes) / len(recent_sizes)

                                if size_variation / avg_size < 0.05:
                                    save_detection = False
                                    current_iou = 0.0

                                    if track_id in tracked_objects:
                                        prev_roi = tracked_objects[track_id]
                                        current_iou = compute_iou(
                                            (x1, y1, x2, y2),
                                            (prev_roi[0], prev_roi[1], prev_roi[2], prev_roi[3])
                                        )
                                        
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
                                        print("befire writing to csv file")
                                        # exit()
                                        # Save detection info to CSV
                                        save_detection_info(csv_file, [
                                            date, timestamp, class_name, track_id, confidence,
                                            current_iou, area_growth_rate, x1, y1, x2, y2, roi_path
                                        ])



                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, 
                                f"{class_name} {track_id} ({confidence:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            results_np = process_frame(frame, numberplate_model)

            if results_np[0].boxes is not None:
                for box in results_np[0].boxes:
                    # Extract bounding box and class info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id) if box.id is not None else -1
                    confidence = float(box.conf[0])
                    class_name = numberplate_model.names[int(box.cls[0])]

                    if confidence >= CONFIDENCE_THRESHOLD and (int(box.cls[0]) + 1) in selected_numberplate_classes:
                        current_area = (x2 - x1) * (y2 - y1)
                        if track_id not in object_buffers:
                            object_buffers[track_id] = deque(maxlen=BUFFER_SIZE)
                            object_max_areas[track_id] = current_area
                            object_entry_frames[track_id] = frame_count

                        if current_area > object_max_areas[track_id]:
                            object_max_areas[track_id] = current_area


                        initial_area = object_buffers[track_id][0][4] if object_buffers[track_id] else current_area
                        area_growth_rate = (current_area - initial_area) / initial_area if initial_area > 0 else 0

                        object_buffers[track_id].append((x1, y1, x2, y2, current_area, frame_count, confidence))


                        if len(object_buffers[track_id]) == BUFFER_SIZE:
                            recent_sizes = [b[4] for b in list(object_buffers[track_id])[-STABLE_THRESHOLD:]]
                            size_variation = max(recent_sizes) - min(recent_sizes)
                            avg_size = sum(recent_sizes) / len(recent_sizes)

                            if size_variation / avg_size < 0.05:
                                save_detection = False
                                current_iou = 0.0


                                if track_id in tracked_objects:
                                    prev_roi = tracked_objects[track_id]
                                    current_iou = compute_iou(
                                        (x1, y1, x2, y2),
                                        (prev_roi[0], prev_roi[1], prev_roi[2], prev_roi[3])
                                    )


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




                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Different color for number plates
                        cv2.putText(frame, 
                                f"{class_name} {track_id} ({confidence:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



        cap.release() 
            
    cap.release()
    return {"message": "Detection completed", "results": results}
