
import json
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import requests
import threading

# import logging
#   # Patches the socket, threading, and other parts for eventlet

# app = Flask(__name__)
# socketio = SocketIO(app)


DJANGO_Data_API_URL = "http://localhost:8000/api/Upload_Data/"
Django_Response_Api = "http://localhost:8000/api/Response_output/"
Django_Video_Summarization_Api = "http://localhost:8000/api/Video_Summarization/"
Django_Video_Translation_Api = "http://localhost:8000/api/Text_Translation/"
Django_Video_Transcription_Api = "http://localhost:8000/api/Audio_Transcription/"
Django_Fake_Video_Api = "http://localhost:8000/api/Deep_Video_Detection/"
Django_Video_Converter_Api = "http://localhost:8000/api/Video_Converter/"
Django_Video_Compresser_Api = "http://localhost:8000/api/Video_Compresser/"

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import json
from pathlib import Path
from flask import current_app as app

app = Flask(__name__)
socketio = SocketIO(app)

script_dir = Path(__file__).resolve().parent  # Path of the current script

# Get the parent directory two levels above the script's directory
parent_dir = script_dir.parent  # Go up two levels from the script's directory

# Define the relative path to the shared storage directory
converted_path = 'shared_storage/Converted_videos'

compress_path = 'shared_storage/Compress_videos'

# Combine the parent directory with the relative path
shared_storage_dir_for_converted = parent_dir / converted_path

shared_storage_dir_for_compresser = parent_dir / compress_path




@app.route('/')
def index():
 # This will contain the hardcoded users data
    return render_template('index.html')


@app.route('/Model_output', methods=['POST', 'GET'])
def model_output():
    if request.method == 'POST':
        data = request.get_json()
        if not data or "rag_query" not in data:
            return jsonify({"error": "Query is required"}), 400

        user_query = data["rag_query"]
        print("User query:", user_query)

        # ✅ Send JSON data correctly
        try:
            response = requests.post(Django_Response_Api, json={"query": user_query}, headers={"Content-Type": "application/json"})
            print("response", response)
            if response.status_code != 200:
                return jsonify({"error": f"Django API error: {response.status_code}"}), response.status_code

            response_data = response.json()  # ✅ Extract JSON properly

            print("Response data", response_data)
            
        except requests.exceptions.RequestException as e:
            return jsonify({"error": "Failed to connect to the Django API"}), 500

        return response.json(), response.status_code  # ✅ Send JSON response properly

    return render_template('rag_result.html', response=None)



@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'bookdata' not in request.files:
        return jsonify({"error": "No Data file provided"}), 400

    file = request.files['bookdata']
    files = {'bookdata': (file.filename, file.stream, file.mimetype)}

    response = requests.post(DJANGO_Data_API_URL, files=files)

    try:
        return response.json(), response.status_code
    except requests.exceptions.JSONDecodeError:
        return jsonify({"error": "Invalid response from server"}), 500


@app.route('/video_summarization', methods = ['POST','GET'])
def video_summarization():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return jsonify({"error": "No Video file provided"}), 400

        file = request.files['video_file']
        files = {'video_file': (file.filename, file.stream, file.mimetype)}

        response = requests.post(Django_Video_Summarization_Api, files=files)

        try:
            return response.json(), response.status_code
        except requests.exceptions.JSONDecodeError:
            return jsonify({"error": "Invalid response from server"}), 500

    return render_template('video_summarization.html', response=None)




@app.route('/translation', methods=['POST'])
def translation():
    if request.method == 'POST':
        if request.content_type == "application/json":
            data = request.get_json()
            if not data or "type" not in data:
                return jsonify({"error": "Invalid JSON data"}), 400

            data_type = data.get('type')

            if data_type == "text":
                text_query = data.get("text")
                src_language = data.get("src_language")
                tgt_language = data.get("tgt_language")

                if not text_query or not src_language or not tgt_language:
                    return jsonify({"error": "Missing required text fields"}), 400

                try:
                    response = requests.post(
                        Django_Video_Translation_Api,
                        json={
                            "src_lang": src_language,
                            "tgt_lang": tgt_language,
                            "text_for_translation": text_query,
                            "type": "text"
                        },
                        headers={"Content-Type": "application/json"}
                    )

                    if response.status_code != 200:
                        return jsonify({"error": f"Django API error: {response.status_code}"}), response.status_code

                    return response.json(), response.status_code

                except requests.exceptions.RequestException:
                    return jsonify({"error": "Failed to connect to the Django API"}), 500

        # Handle File Upload
        elif "translation_file" in request.files and request.form.get("type") == "file":
            file = request.files['translation_file']
            src_language_file = request.form.get('src_lang_file')
            tgt_language_file = request.form.get('tgt_lang_file')
            if not file:
                return jsonify({"error": "No file uploaded"}), 400

            files = {'translation_file': (file.filename, file.stream, file.mimetype)}
            response = requests.post(Django_Video_Translation_Api, files=files, 
                                     data={"type": "file",'src_lang_file':src_language_file,
                                           'tgt_lang_file':tgt_language_file})

            if response.status_code != 200:
                return jsonify({"error": f"Django API error: {response.status_code}"}), response.status_code

            return response.json(), response.status_code

        return jsonify({"error": "Invalid request"}), 400

    return render_template('translation.html')


import os
from flask import send_from_directory

@app.route('/download/<filename>')
def download_video(filename):
    directory = shared_storage_dir_for_converted  # Use the constructed path here
    print(f"Looking for {filename} in directory {directory}")

    # Check if the file exists
    file_path = directory / filename
    if not file_path.exists():
        print(f"File {filename} not found at {file_path}")
        return "File not found", 404

    return send_from_directory(directory, filename, as_attachment=True)





@app.route('/download_compress/<filename>')
def download_compress_video(filename):
    directory = shared_storage_dir_for_compresser  # Use the constructed path here
    print(f"Looking for {filename} in directory {directory}")

    # Check if the file exists
    file_path = directory / filename
    if not file_path.exists():
        print(f"File {filename} not found at {file_path}")
        return "File not found", 404

    return send_from_directory(directory, filename, as_attachment=True)




@app.route('/Video_Converter', methods=['POST'])
def video_converter():
    if request.method == 'POST':
        if 'video_converter_file' not in request.files:
            return jsonify({"error": "No Video file provided"}), 400

        file = request.files['video_converter_file']
        video_format = request.form.get('video_format')
        print("video format", video_format)
        files = {'video_converter_file': (file.filename, file.stream, file.mimetype)}
        print('files', files)
        response = requests.post(Django_Video_Converter_Api, files=files, data={"type":video_format})

        try:
            result = response.json()
            if response.status_code == 200 and "result" in result:
                file_path = result["result"].get('file_path')
                filename = result["result"].get('filename')
                download_url = f"/download/{filename}"
                download_url2 = request.host_url + 'download/' + result["result"]["filename"]
                print("download_url2", download_url2)

                return jsonify({
                        "status": "Completed",
                        "filename": filename,
                        "download_url": download_url2
                    })

                # return response.json(), response.status_code
        
        except requests.exceptions.JSONDecodeError:
            return jsonify({"error": "Invalid response from server"}), 500








@app.route('/Video_Compresser', methods=['POST'])
def video_compresser():
    if request.method == 'POST':
        if 'video_compresser_file' not in request.files:
            return jsonify({"error": "No Video file provided"}), 400

        file = request.files['video_compresser_file']
        compress_rate = request.form.get('compress_rate')
        print("compress_rate", compress_rate)
        files = {'video_compresser_file': (file.filename, file.stream, file.mimetype)}
        print('files', files)
        response = requests.post(Django_Video_Compresser_Api, files=files, data={"compress_rate":compress_rate})
        try:
                result = response.json()
                if response.status_code == 200 and "result" in result:
                    file_path = result["result"].get('file_path')
                    filename = result["result"].get('filename')
                    download_url2 = request.host_url + 'download_compress/' + result["result"]["filename"]
                    print("download_url2", download_url2)

                    return jsonify({
                            "status": "Completed",
                            "filename": filename,
                            "download_url": download_url2
                        })

                    # return response.json(), response.status_code
            
        except requests.exceptions.JSONDecodeError:
            return jsonify({"error": "Invalid response from server"}), 500



correction_list = []

@app.route('/save_correction', methods=['POST'])
def save_correction_api():
    data = request.get_json()
    print("aall lr data", data)
    correction_list.append({
        "translated": data["translated"],
        "corrected": data["corrected"],

    })

    print("list", correction_list)

    return jsonify({"message": "Correction saved", "corrections": correction_list}), 200


@app.route('/transcription', methods =['POST','GET'])
def transcription():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return jsonify({"error": "No Video file provided"}), 400
        trans_language = request.form['trans_language']
        print('trans_language',trans_language)
        file = request.files['audio_file']
        files = {'audio_file': (file.filename, file.stream, file.mimetype)}

        data = {'trans_language': trans_language}
        
        response = requests.post(Django_Video_Transcription_Api, files=files, data = data)

        try:
            return response.json(), response.status_code
        except requests.exceptions.JSONDecodeError:
            return jsonify({"error": "Invalid response from server"}), 500
    
    return render_template('transcription.html')



@app.route('/fake_video', methods =['POST','GET'])
def fake_video():
    print("fake video")
    if request.method == 'POST':
        print('inside post')
        if 'fake_video_file' not in request.files:
            return jsonify({"error": "No Video file provided"}), 400

        file = request.files['fake_video_file']
        files = {'fake_video_file': (file.filename, file.stream, file.mimetype)}

        response = requests.post(Django_Fake_Video_Api, files=files)

        try:
            return response.json(), response.status_code
        except requests.exceptions.JSONDecodeError:
            return jsonify({"error": "Invalid response from server"}), 500
    
    return render_template('fakevideo.html')

if __name__ == '__main__':
    socketio.run(app, port=8585)
    # socketio.run(app, host='0.0.0.0', port=8585, debug=True)
