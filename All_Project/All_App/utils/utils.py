from uuid import uuid4
from elasticsearch import Elasticsearch
import os
import cv2
import numpy as np
import ffmpeg
import logging




es = Elasticsearch("http://localhost:9200")  # Update with your Elasticsearch host if needed

# Define Elasticsearch indices
INDEX_PROCESSED_FILES = "processed_files"
INDEX_DOCUMENTS = "documents"

# Create indices if they do not exist
if not es.indices.exists(index=INDEX_PROCESSED_FILES):
    es.indices.create(index=INDEX_PROCESSED_FILES)

if not es.indices.exists(index=INDEX_DOCUMENTS):
    es.indices.create(index=INDEX_DOCUMENTS)


def insert_documents_to_es(documents, embeddings_model):
    for doc in documents:
        doc_id = str(uuid4())  # Generate a unique ID
        embedding = embeddings_model.embed_query(doc.page_content)

        if not embedding or len(embedding) == 0:
            print(f"Skipping invalid embedding for document: {doc_id}")
            continue  # Skip invalid embeddings

        es.index(
            index=INDEX_DOCUMENTS,
            id=doc_id,
            document={
                "metadata": doc.metadata,
                "content": doc.page_content,
                "embedding": embedding,
            }
        )
        print(f"Inserted document with ID {doc_id}: {doc.page_content[:10]}...")

    print(f"Inserted {len(documents)} documents into the '{INDEX_DOCUMENTS}' index.")

def file_already_processed(filename):
    """Check if the file is already processed in Elasticsearch."""
    query = {"query": {"term": {"filename.keyword": filename}}}
    result = es.search(index=INDEX_PROCESSED_FILES, body=query)
    return result["hits"]["total"]["value"] > 0


def mark_file_as_processed(filename):
    """Mark the file as processed in Elasticsearch."""
    es.index(index=INDEX_PROCESSED_FILES, document={"filename": filename})







def video_summarize(file_path):
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






def Audio_Video_Transcription(Media_Path):
    import whisper
    model = whisper.load_model("tiny")
    result = model.transcribe(Media_Path)
    result = result["text"]
    return result



def get_video_duration(file_path):
    try:
        probe = ffmpeg.probe(str(file_path))
        duration = float(probe['format']['duration'])
        return round(duration, 2)
    except Exception as e:
        print(f"Error getting video duration: {str(e)}")  # Debugging
        # logger.error(f"Error getting video duration: {str(e)}")
        return None