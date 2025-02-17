import os
from groq import Groq

# Initialize Groq client (ensure API key is set in environment variables)
client = Groq(api_key="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")

# Path to the audio file
filename = r"C:\Users\Msawaiz10\Downloads\biden1_iris3 (online-video-cutter.com).mp3"

# Open and read the file
with open(filename, "rb") as file:
    transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),  # Read file content
        model="whisper-large-v3",       # Use Whisper for transcription
        response_format="verbose_json", # Get detailed JSON output
    )

# Print only the transcribed text
print("Transcription:", transcription.text)
