from celery import shared_task, group, current_task
import time
from pydub import AudioSegment
import whisper
import requests
import subprocess
from django.conf import settings
import os
from datetime import datetime
from docx2pdf import convert
import PyPDF2
from gtts import gTTS
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import pymupdf
import moviepy.editor as mp
import whisperx
import torch  # Import torch directly
import gc
from openai import OpenAI
import time
import fitz  # PyMuPDF


apikey = "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"


client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=apikey
)

@shared_task
def Whisper_audio_check(input_path, converted_path):

            # Debug prints
    print(f"Received Input Path: {input_path}")
    print(f"Received Output Path: {converted_path}")


    video = mp.VideoFileClip(input_path)
    start_time = time.time()
    output_path = converted_path    
    video.audio.write_audiofile(output_path)
    device = "cpu"
    batch_size = 16
    compute_type = "int8"

    model = whisperx.load_model("tiny", device, compute_type=compute_type)

    # Load audio file and transcribe
    audio = whisperx.load_audio(output_path)
    result = model.transcribe(audio, batch_size=batch_size)

    # Collect all text from segments
    transcribed_text = ' '.join([segment['text'] for segment in result['segments']])

    print('transcrrube_text', transcribed_text)



    if not transcribed_text:
        print('text khali ha')
        response_text = "No ingredients found in the transcribed text."
    else:
        print('in else')
        messages = [
            {"role": "user", "content": f"{transcribed_text} Extract the ingredients from above text."}
        ]


    
    completion = client.chat.completions.create(
    model="meta/llama3-70b-instruct",
    messages=messages,
    temperature=0.5,
    top_p=1,
    max_tokens=1024,
    stream=True
    )


    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content

    print("Extracted Ingredients:", response_text)



    return {
        "audio_path": output_path,
        "transcribed_text": transcribed_text,
        "response_text" : response_text
    }

