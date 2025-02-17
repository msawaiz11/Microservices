from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import os




from django.conf import settings
from All_App.utils.utils import (file_already_processed, mark_file_as_processed, 
                                 insert_documents_to_es,video_summarize,Text_For_Translation,Audio_Video_Transcription)
from All_App.utils.rag_output import retrieve_and_generate_response
from All_App.tasks import Audio_Video_Transcription_celery
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader, UnstructuredImageLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import cv2 # pip install opencv-python
import numpy as np # pip install numpy

from All_App.llama_models import Models
from uuid import uuid4
from celery.result import AsyncResult
import time
# Load environment variables
load_dotenv()

models = Models()
embeddings = models.embeddings_ollama
llm = models.model_ollama



# Create your views here.



chunk_size = 1000
chunk_overlap = 50
check_interval = 10


LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredExcelLoader,
}




class RagData(APIView):
    parser_classes = (MultiPartParser, FormParser)
    def post(self, request):
        if "bookdata" not in request.FILES:
            return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)
        all_files = request.FILES['bookdata']
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join(media_dir, all_files.name)
        with open(file_path, 'wb+') as destination:
            for chunk in all_files.chunks():
                destination.write(chunk)

        filename = os.path.basename(file_path)

        if file_already_processed(filename):
            print(f"Skipping already processed file: {filename}")
            return Response({"message": "File already has been processed", "file_path": file_path}, status=status.HTTP_201_CREATED)

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in LOADERS:
            print(f"Skipping unsupported file format: {file_path}")
            return Response({"message": "File format not supported", "file_path": file_path}, status=status.HTTP_201_CREATED)

        print(f"Processing file: {file_path}")
        loader = LOADERS[file_ext](file_path)
        loaded_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", " ", ""]
            )

        documents = text_splitter.split_documents(loaded_documents)

        insert_documents_to_es(documents, embeddings)
        mark_file_as_processed(filename)
        
        return Response({"message": "File uploaded successfully", "file_path": file_path}, status=status.HTTP_201_CREATED)







class RagData_Generation(APIView):
    def post(self, request):
        user_query = request.data.get("query", "")
        print("user_query", user_query)
        if not user_query:
            return Response({"error": "Query parameter is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Call the retrieval function from the separate file
        result = retrieve_and_generate_response(user_query)
        print("result", result)

        return Response({"response": result}, status=status.HTTP_200_OK)


class Video_Summarization(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if "video_file" not in request.FILES:
            return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)
        all_files = request.FILES['video_file']
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        print("typemedia", type(media_dir))
        os.makedirs(media_dir, exist_ok=True)  # Ensure the directory exists

        file_path = os.path.join(media_dir, all_files.name)
        print("filepath", type(file_path))
        with open(file_path, 'wb+') as destination:
            for chunk in all_files.chunks():
                destination.write(chunk)
        filename = os.path.basename(file_path)
        print("filename", filename)
        data = video_summarize(file_path)        
        print("data", data)

        return Response({"message": "Video Summarized successfully", "file_path": file_path}, status=status.HTTP_201_CREATED)



class Text_Translation(APIView):
    def post(self, request):
        text = request.data.get("text_for_translation", "")
        print("text_query", text)
        if not text:
            return Response({"error": "Query parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        Translated_Text = Text_For_Translation(text)
        print('Translated_Text',Translated_Text)
        return Response({"response": Translated_Text}, status=status.HTTP_200_OK)



class Audio_Transcription(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if "audio_file" not in request.FILES:
            return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Save the uploaded audio file
        all_files = request.FILES['audio_file']
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, all_files.name)
        with open(file_path, 'wb+') as destination:
            for chunk in all_files.chunks():
                destination.write(chunk)

        # Trigger Celery task for transcription
        task = Audio_Video_Transcription_celery.delay(file_path)
        
        # Poll for the task result (not ideal for long tasks, but works for shorter ones)
        task_result = AsyncResult(task.id)
        print('task_result', task_result)
        while task_result.state in ["PENDING", "STARTED"]:
            time.sleep(2)  # Wait for 2 seconds before checking again
            task_result = AsyncResult(task.id)


        # Return result once task is complete
        if task_result.state == "SUCCESS":
            return Response({"status": "Completed", "result": task_result.result}, status=status.HTTP_200_OK)
        elif task_result.state == "FAILURE":
            return Response({"status": "Failed", "error": str(task_result.info)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"status": task_result.state}, status=status.HTTP_202_ACCEPTED)