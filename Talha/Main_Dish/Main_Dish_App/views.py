from django.shortcuts import render
# views.py

import os
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .tasks import Whisper_audio_check
from Main_Dish_App.serializer import ProcessAudioSerializer



@api_view(['POST'])
def process_audio(request):
    if 'input_file' not in request.FILES:
        return Response({"error": "No input file provided."}, status=status.HTTP_400_BAD_REQUEST)
    
    input_file = request.FILES['input_file']
    output_file_name = f"output_{input_file.name.rsplit('.', 1)[0]}.mp3"  # Example output file name

    # Save the uploaded file
    input_path = os.path.join(settings.MEDIA_ROOT, input_file.name)
    with open(input_path, 'wb+') as destination:
        for chunk in input_file.chunks():
            destination.write(chunk)
    
    # Generate output path
    output_path = os.path.join(settings.MEDIA_ROOT, output_file_name)
    
    # Debug prints
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    
    # Run the Celery task synchronously
    task_result = Whisper_audio_check.delay(input_path, output_path)
    task_result.get()  # Wait for the task to complete
    
    # Get the result
    result = task_result.result
    response_text = result.get('response_text', 'No response text found.')

    
    return Response({
        # "task_id": task_result.id,
        "response_text": response_text,
        # "status": "Task completed!"
    }, status=status.HTTP_200_OK)




def index(request):
    return render(request, "index.html")