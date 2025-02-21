from celery import shared_task
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
tokenizer = AutoTokenizer.from_pretrained(
    r"E:\t_model\models--facebook--nllb-200-distilled-1.3B\snapshots\7be3e24664b38ce1cac29b8aeed6911aa0cf0576"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    r"E:\t_model\models--facebook--nllb-200-distilled-1.3B\snapshots\7be3e24664b38ce1cac29b8aeed6911aa0cf0576"
)


import cv2
import os
import numpy as np
import ffmpeg
import time
import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from django.conf import settings
from PIL import Image
from realesrgan_ncnn_py import Realesrgan

from datetime import datetime, timedelta
from All_App.utils.feekvideo import load_detection_model, predict_frames, MODEL_PATH
from All_App.utils.rag_output import multi_query_retriever, combine_docs_chain
from All_App.utils.extraction_functions import All_Extraction
from All_App.utils.utils import get_video_duration
from All_App.utils.object_detection_new_roi import (get_timestamp, process_frame,numberplate_model,
                                                    CONFIDENCE_THRESHOLD, yolo_model, config,compute_iou, save_detection)



@shared_task
def Audio_Video_Transcription_celery(Media_Path, language):
    import whisper
    model = whisper.load_model("tiny")
    result = model.transcribe(Media_Path,language=language)
    result = result["text"]
    return result


@shared_task
def Video_Converter_Celery(file_path, video_format):
    print("file_path", file_path)
    file_path = Path(file_path)
    start_time = time.time()
    duration = get_video_duration(file_path)
    print('video durariont', duration)
  

    output_filename = f"{file_path.stem}.{video_format}"


    script_dir = Path(__file__).resolve().parent  # Path of the current script

    # Get the parent directory two levels above the script's directory
    parent_dir = script_dir.parent.parent  # Go up two levels from the script's directory

    # Define the relative path from the parent directory
    relative_path = 'shared_storage/Converted_videos'

    # Combine the parent directory with the relative path
    shared_storage_dir = parent_dir / relative_path

    # Create the directory if it doesn't exist
    shared_storage_dir.mkdir(parents=True, exist_ok=True)

    # Create the full output path
    output_path = shared_storage_dir / output_filename 

    # Continue your FFmpeg conversion logic here...
    print(f"Output path: {output_path}")


    try:
        # Set up FFmpeg command
        stream = (ffmpeg.input(str(file_path)).output(str(output_path),acodec='aac' if video_format == 'mp4' else 'copy',
                    vcodec='libx264' if video_format in ['mp4', 'avi'] else 'copy',
                    strict='experimental')
        )

        # Run the conversion
        
        logger.info("Starting FFmpeg conversion...")
        ffmpeg.run(stream, capture_stderr=True, overwrite_output=True)
        logger.info("FFmpeg conversion completed")

        # Verify the output file
        if not output_path.exists():
            raise Exception("Output file was not created")
        
        if output_path.stat().st_size == 0:
            raise Exception("Output file is empty")

        logger.info(f"Output file size: {output_path.stat().st_size} bytes")

          # Clean up input file
        os.remove(file_path)
        logger.info("Cleaned up input file")
        
        # Calculate elapsed time
        elapsed_time = round(time.time() - start_time, 2)
        
        return {
            "filename": output_filename,
            "duration": duration,
            "conversion_time": elapsed_time,
            "file_path": f"{output_path}/{output_filename}",
        }

    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error: {error_message}")





@shared_task
def Video_Compress_Celery(file_path, compress_rate):
    file_path = Path(file_path)
    start_time = time.time()
    output_filename = f"{file_path.stem}_compressed.mp4"

    script_dir = Path(__file__).resolve().parent  # Path of the current script

    # Get the parent directory two levels above the script's directory
    parent_dir = script_dir.parent.parent  # Go up two levels from the script's directory

    # Define the relative path from the parent directory
    relative_path = 'shared_storage/Compress_videos'

    # Combine the parent directory with the relative path
    shared_storage_dir = parent_dir / relative_path

    # Create the directory if it doesn't exist
    shared_storage_dir.mkdir(parents=True, exist_ok=True)

    # Create the full output path
    output_path = shared_storage_dir / output_filename     
    compress_rate = float(compress_rate)
    try:
        # Get video information
        probe = ffmpeg.probe(str(file_path))
        logger.info(f"Probe result: {probe}")
        
        # Set default compression settings based on compression_rate
        crf_value = int(18 + (compress_rate / 100.0) * 10)
        
        # Build FFmpeg command with proper parameters
        stream = (
            ffmpeg
            .input(str(file_path))
            .output(
                str(output_path),
                acodec='aac',
                vcodec='libx264',
                preset='medium',
                crf=str(crf_value)
            )
        )
        
        # Run the compression
        logger.info(f"Starting FFmpeg compression with CRF {crf_value}...")
        ffmpeg.run(stream, capture_stderr=True, overwrite_output=True)
        logger.info("Compression completed")
        
        # Verify the output file
        if not output_path.exists():
            raise Exception("Output file was not created")
        
        if output_path.stat().st_size == 0:
            raise Exception("Output file is empty")
            
        # Get file sizes for comparison
        original_size = os.path.getsize(str(file_path))
        compressed_size = os.path.getsize(str(output_path))
        compression_ratio = ((original_size - compressed_size) / original_size) * 100
        
        # Clean up input file
        os.remove(file_path)
        
        # Calculate elapsed time
        elapsed_time = round(time.time() - start_time, 2)
        
        return {
            "filename": output_filename,
            "original_size": f"{original_size / (1024*1024):.2f} MB",
            "compressed_size": f"{compressed_size / (1024*1024):.2f} MB",
            "compression_ratio": f"{compression_ratio:.1f}%",
            "compression_time": elapsed_time,
            "file_path": f"{output_path}/{output_filename}",
        }
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error: {error_message}")

        # if os.path.exists(str(file_path)):
        #     os.remove(file_path)
        # return Response(
        #     status_code=500,
        #     content={"error": f"Compression error: {error_message}"}
        # )



@shared_task
def Text_Translation_celery(data_type,src_lang,tgt_lang,text_query,file_path):
    if data_type == "text":
        print("inside text query")
        translator = pipeline("translation",model=model,tokenizer=tokenizer,src_lang=src_lang,tgt_lang=tgt_lang, device='cpu')
        translated_chunk = translator(text_query, max_length=400)[0]["translation_text"]
        return translated_chunk

    elif data_type == "file":
        print("filepath", file_path)
        data = All_Extraction(file_path)
        print("data", data)
        translator = pipeline("translation",model=model,tokenizer=tokenizer,src_lang=src_lang,tgt_lang=tgt_lang, device=1)
        
        for index, paragraph in enumerate(data, start=1):
            translated_chunk = translator(paragraph, max_length=400)[0]["translation_text"]
            print("tranaslted chunk", translated_chunk)
        return translated_chunk
        




@shared_task
def analyze_video(video_path,frame_skip=30):
    model = load_detection_model(MODEL_PATH)
    if model is None:
        print("Exiting due to model loading error.")
        return "Error: Model could not be loaded"


    video = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    if not video.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return frames

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if count % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # Resize for model input
            frames.append(frame)

        count += 1



    frames = np.array(frames)
    predictions = predict_frames(model, frames)

    if len(predictions) == 0:
        print("No predictions made. Exiting.")
        return "Error: No predictions made"


    # Summarize detection results
    fake_count = np.sum(np.array(predictions) > 0.5)  # Count fakes
    real_count = len(predictions) - fake_count        # Count reals

    # Return a final decision based on majority prediction
    if fake_count > real_count:
        results = "Fake Video"
        # return results, fake_count, real_count
        return results, int(fake_count), int(real_count)

    else:
        results = "Real Video"
        # return results, fake_count, real_count
        return results, int(fake_count), int(real_count)





@shared_task
def video_summarize_celery(file_path):
    cap = cv2.VideoCapture(file_path)  # Open video file

    if not cap.isOpened():
        return {"error": "Failed to open video file"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Width and Height:", width, height)

    threshold = 20.0

    output_path = os.path.join("media", "final.mp4")  # Adjust path as needed
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        writer.release()
        return {"error": "Failed to read first frame"}

    a, b, c = 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:  # Exit loop when no more frames
            break

        if (np.sum(np.abs(frame - prev_frame)) / np.size(frame)) > threshold:
            writer.write(frame)
            prev_frame = frame
            a += 1
        else:
            prev_frame = frame
            b += 1

        c += 1

    cap.release()
    writer.release()

    return {"output_path": output_path, "frames_written": a, "frames_skipped": b, "total_frames": c}





@shared_task
def retrieve_and_generate_response_celery(user_query):
    """Retrieves relevant documents and generates a response."""
    # Debug: Print the user query
    print("User Query:", user_query)

    docs = multi_query_retriever.get_relevant_documents(user_query)

    # Retrieve documents using the multi_query_retriever's invoke method
    # docs = multi_query_retriever.invoke({"input": user_query})

    # Debug: Print the number of documents retrieved
    print("Number of Documents Retrieved:", len(docs))

    # if not docs:
    #     return "No relevant documents found."

    # Call the document processing chain
    result = combine_docs_chain.invoke({"input": user_query, "context": docs})
    print("result", result)
    return result


tracked_objects = {}  # Stores final saved ROIs
object_buffers = {} 
BUFFER_SIZE = 10 
object_max_areas = {}  # Stores maximum area seen for each tracked object
object_entry_frames = {}
AREA_GROWTH_THRESHOLD = 0.2
STABLE_THRESHOLD = 3







import cv2
import os
from datetime import datetime
from collections import deque
from celery import shared_task
import shutil

# Define missing variables (Ensure they are set correctly in your real implementation)
object_buffers = {}
object_max_areas = {}
object_entry_frames = {}

@shared_task
def Object_Detection_Celery(video_path, check_box_names, selected_classes):
    print("Selected classes:", selected_classes)
    print("Checkbox names:", check_box_names)
    detections_list = []        

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
        selected_numberplate_classes = None
        # print("Processing other objects...")
        # print("selected classes", selected_classes)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            date, timestamp = get_timestamp(frame_count, fps, video_start_time)




            frame_detections = process_frame(frame, frame_count, date, timestamp, yolo_model, numberplate_model,
                                        tracked_objects, object_buffers, object_max_areas, config,
                                        output_folder, csv_file, selected_classes,selected_numberplate_classes, video_path)



            # print("frame_detections",frame_detections)

            detections_list.extend(frame_detections)

            # print("detelction list", detections_list)


           

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
            
            frame_detections = process_frame(frame, frame_count, date, timestamp, yolo_model, numberplate_model,
                                        tracked_objects, object_buffers, object_max_areas, config,
                                        output_folder, csv_file,selected_classes,selected_numberplate_classes, video_path)

            # print("frame_detections",frame_detections)

            detections_list.extend(frame_detections)

            # print("detelction list", detections_list)
        
        cap.release() 
            
    cap.release()
    try:
        shutil.rmtree(output_folder)
        print(f"Successfully removed folder: {output_folder}")
    except Exception as e:
        print(f"Error while deleting folder {output_folder}: {e}")

    return {"message": "Detection completed", "results": detections_list}








@shared_task
def Object_Enhance_Celery(file_path, file_type):
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent.parent
    relative_path = 'shared_storage/Enhance_File'
    shared_storage_dir = parent_dir / relative_path

    # Ensure the output directory exists
    shared_storage_dir.mkdir(parents=True, exist_ok=True)

    file_name, file_ext = os.path.splitext(os.path.basename(file_path))
    enhanced_file_name = f"{file_name}_enhanced{file_ext}"  # ✅ Define the name for both images and videos
    enhanced_file_path = str(shared_storage_dir / enhanced_file_name)  # ✅ Ensure it is defined before use
    realesrgan = Realesrgan(gpuid=0, model=0)

    if file_type == "image_file":
        
        with Image.open(file_path) as image:
            image = realesrgan.process_pil(image)
            # Save the image correctly
            image.save(str(enhanced_file_path), quality=95)  # Convert Path object to string

        print("Enhanced file saved at:", enhanced_file_path)

        try:
            os.remove(file_path)
            print(f"Deleted original file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    elif file_type == "video_file":  
        # Define target resolution or scale factor
        target_resolution = (1920, 1080)  # Set to (width, height) or "2x"

        # Read input video
        cap = cv2.VideoCapture(file_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video FPS
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if video_width == 0 or video_height == 0:
            print("\n❌ Error: Unable to read video file! Check if the file is corrupted.\n")
            exit(1)

        # Determine scaling factor
        aspect_ratio = video_width / video_height

        if isinstance(target_resolution, str):  # Handles "2x", "3x" scaling cases
            scale_factor = int(target_resolution[0])
            final_width = video_width * scale_factor
            final_height = video_height * scale_factor
        else:
            final_width, final_height = target_resolution
            scale_factor = max(final_width / video_width, final_height / video_height)

        # Ensure final resolution is even
        while (int(video_width * scale_factor) % 2 != 0) or (int(video_height * scale_factor) % 2 != 0):
            scale_factor += 0.01

        # Final adjusted width and height
        final_width = int(video_width * scale_factor)
        final_height = int(video_height * scale_factor)

        # Define video writer for output
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
        out = cv2.VideoWriter(enhanced_file_path, fourcc, fps, (final_width, final_height))

        print(f"Processing video frames... Upscaling to {final_width}x{final_height}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            enhanced_image = realesrgan.process_pil(image)
            enhanced_frame = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
            enhanced_frame = cv2.resize(enhanced_frame, (final_width, final_height), interpolation=cv2.INTER_CUBIC)
            out.write(enhanced_frame)

        cap.release()
        out.release()
        print("Enhanced video saved at:", enhanced_file_path)

    # ✅ Delete the original file after processing
    try:
        os.remove(file_path)
        print(f"Deleted original file: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

    return {
        "filename": enhanced_file_name,
        "file_path": enhanced_file_path
    }