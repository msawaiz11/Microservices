from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import os
import json
import pika
from django.conf import settings

from All_App.utils.utils import (file_already_processed, mark_file_as_processed, 
                                 insert_documents_to_es)


from All_App.tasks import (Audio_Video_Transcription_celery, Text_Translation_celery,
                            analyze_video, video_summarize_celery ,
                            retrieve_and_generate_response_celery,
                              Video_Converter_Celery, Video_Compress_Celery, 
                              Object_Detection_Celery,Object_Enhance_Celery, Crowd_Detection_Celery,
                              Video_Converter_Celery, Video_Compress_Celery, Object_Detection_Celery,Object_Enhance_Celery)


##### rag libraries #####
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader, UnstructuredImageLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from All_App.llama_models import Models
# from All_App.utils.rag_output import retrieve_and_generate_response
load_dotenv()

models = Models()
embeddings = models.embeddings_ollama
llm = models.model_ollama


chunk_size = 1000
chunk_overlap = 50
check_interval = 10


LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredExcelLoader,
}




##### rag libraries #####

import cv2 # pip install opencv-python
import numpy as np # pip install numpy


from uuid import uuid4
from celery.result import AsyncResult
import time
# Load environment variables




# Create your views here.






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
        task = retrieve_and_generate_response_celery.delay(user_query)
        task_result = AsyncResult(task.id)
        while task_result.state in ["PENDING", "STARTED"]:
            time.sleep(2)  # Wait for 2 seconds before checking again
            task_result = AsyncResult(task.id)


        # Return result once task is complete
        if task_result.state == "SUCCESS":
            return Response({"status": "Completed", "result": task_result.result}, status=status.HTTP_200_OK)
        elif task_result.state == "FAILURE":
            return Response({"status": "Failed", "error": str(task_result.info)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



        return Response({"response": task_result.result}, status=status.HTTP_200_OK)


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
        task = video_summarize_celery.delay(file_path)        

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

        return Response({"message": "Video Summarized successfully", "file_path": file_path}, status=status.HTTP_201_CREATED)



class Text_Translation(APIView):
    def post(self, request):
        src_lang = request.data.get("src_lang","")
        tgt_lang = request.data.get("tgt_lang","")
        data_type = request.data.get("type", "")

        print("sssss", data_type)

        if data_type == "text":

            text = request.data.get("text_for_translation", "")
            print("text_query", text)
            if not text:
                return Response({"error": "Query parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
            
            self.send_message_to_rabbitmq(text)
            
            task = Text_Translation_celery.delay(data_type,src_lang,tgt_lang,text,None)
            
            task_result = AsyncResult(task.id)
            while task_result.state in ["PENDING", "STARTED"]:
                time.sleep(2)  # Wait for 2 seconds before checking again
                task_result = AsyncResult(task.id)


            # Return result once task is complete
            if task_result.state == "SUCCESS":
                return Response({"status": "Completed", "result": task_result.result}, status=status.HTTP_200_OK)
            elif task_result.state == "FAILURE":
                return Response({"status": "Failed", "error": str(task_result.info)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({"status": task_result.state}, status=status.HTTP_202_ACCEPTED)

        elif data_type == "file":
            print("inside the in class api")
            src_lang_file = request.data.get("src_lang_file")
            tgt_lang_file = request.data.get('tgt_lang_file')
            print('ssasdads', src_lang_file, tgt_lang_file)    

            all_files = request.FILES['translation_file']
            media_dir = os.path.join(settings.BASE_DIR, 'media')
            os.makedirs(media_dir, exist_ok=True)
            file_path = os.path.join(media_dir, all_files.name)
            with open(file_path, 'wb+') as destination:
                for chunk in all_files.chunks():
                    destination.write(chunk)

            task = Text_Translation_celery.delay(data_type, src_lang_file, tgt_lang_file, None, file_path)


            task_result = AsyncResult(task.id)
            while task_result.state in ["PENDING", "STARTED"]:
                time.sleep(2)  # Wait for 2 seconds before checking again
                task_result = AsyncResult(task.id)


            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} deleted successfully.")

            # Return result once task is complete
            if task_result.state == "SUCCESS":
                return Response({"status": "Completed", "result": task_result.result}, status=status.HTTP_200_OK)
            elif task_result.state == "FAILURE":
                return Response({"status": "Failed", "error": str(task_result.info)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({"status": task_result.state}, status=status.HTTP_202_ACCEPTED)



    def send_message_to_rabbitmq(self, text):
            """Send a message to RabbitMQ after saving the audio file."""
            # connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
            connection = pika.BlockingConnection(pika.ConnectionParameters(host='127.0.0.1'))

            channel = connection.channel()
            
            # Declare queue (Ensure queue exists before sending message)
            channel.queue_declare(queue='audio_processing')

            # Message to send
            message = json.dumps({"text": text, "status": "uploaded"})
            print(f"Sending message to RabbitMQ: {message}")  # Log the message

            # Publish message
            channel.basic_publish(exchange='', routing_key='audio_processing', body=message)
            
            # Close connection
            connection.close()




class Audio_Transcription(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if "audio_file" not in request.FILES:
            return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Save the uploaded audio file
        all_files = request.FILES['audio_file']
        language = request.data.get('trans_language')
        print('language', language)
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, all_files.name)
        with open(file_path, 'wb+') as destination:
            for chunk in all_files.chunks():
                destination.write(chunk)

        # Trigger Celery task for transcription
        task = Audio_Video_Transcription_celery.delay(file_path, language)
        
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
    




class Video_Compresser(APIView):

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if "video_compresser_file" not in request.FILES:
            return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Save the uploaded audio file
        compress_rate = request.data.get("compress_rate", "")
        print("data type", compress_rate)
        all_files = request.FILES['video_compresser_file']
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, all_files.name)
        with open(file_path, 'wb+') as destination:
            for chunk in all_files.chunks():
                destination.write(chunk)

        task = Video_Compress_Celery.delay(file_path,compress_rate)
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



class Video_Converter_Api(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if "video_converter_file" not in request.FILES:
            return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Save the uploaded audio file
        video_format = request.data.get("type", "")
        print("data type", video_format)
        all_files = request.FILES['video_converter_file']
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, all_files.name)
        with open(file_path, 'wb+') as destination:
            for chunk in all_files.chunks():
                destination.write(chunk)
                
        # exit()
        # Trigger Celery task for transcription
        task = Video_Converter_Celery.delay(file_path,video_format)
        
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








class Deep_Video_Detection(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if "fake_video_file" not in request.FILES:
            return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        all_files = request.FILES['fake_video_file']
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, all_files.name)
        with open(file_path, 'wb+') as destination:
            for chunk in all_files.chunks():
                destination.write(chunk)
       
        task = analyze_video.delay(file_path)
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

        return Response({"message": "File uploasded processed", "file_path": file_path}, status=status.HTTP_201_CREATED)
    





class Object_Detection_Api(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if "object_detection_file" not in request.FILES:
            return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        all_files = request.FILES['object_detection_file']

        # Get the checkbox names and values from the JSON body
        check_box_names = request.data.get("check_box_names", [])
        check_box_values = request.data.get("check_box_values", [])

        print('checkbox names and values:', check_box_names, check_box_values)

        if not check_box_names or not check_box_values:
            return Response({"error": "Missing checkbox names or values"}, status=status.HTTP_400_BAD_REQUEST)

        # Process the file (save to disk)
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, all_files.name)
        with open(file_path, 'wb+') as destination:
            for chunk in all_files.chunks():
                destination.write(chunk)

        # Proceed with the object detection task using the Celery task
        task = Object_Detection_Celery.delay(file_path, check_box_names, check_box_values)
        print('Task started:', task)

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



class Object_Enhance_Api(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if "object_detection_file" not in request.FILES:
            return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        all_files = request.FILES['object_detection_file']
        file_type = request.data.get("type", "")
        all_files = request.FILES['object_detection_file']
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, all_files.name)
        with open(file_path, 'wb+') as destination:
            for chunk in all_files.chunks():
                destination.write(chunk)

        task = Object_Enhance_Celery.delay(file_path, file_type)
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

        return Response({"message": "File uploasded processed", "file_path": file_path}, status=status.HTTP_201_CREATED)

class Crowd_Detection_Api(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if "crowd_detection_file" not in request.FILES:
            return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        all_files = request.FILES['crowd_detection_file']
        file_type = request.data.get("type", "")
        print("file type", file_type)
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, all_files.name)
        with open(file_path, 'wb+') as destination:
            for chunk in all_files.chunks():
                destination.write(chunk)

        task = Crowd_Detection_Celery.delay(file_path, file_type)
        print('task', task)

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

        return Response({"message": "File uploasded processed", "file_path": file_path}, status=status.HTTP_201_CREATED)

