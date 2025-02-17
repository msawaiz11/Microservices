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


from All_App.utils.feekvideo import load_detection_model, predict_frames, MODEL_PATH
from All_App.utils.rag_output import multi_query_retriever, combine_docs_chain
from All_App.utils.extraction_functions import All_Extraction
from All_App.utils.utils import get_video_duration


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
