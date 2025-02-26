
import json
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import requests

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
Django_Object_Detection_Api = "http://localhost:8000/api/Object_Detection_Api/"
Django_Object_Enhance_Api = "http://localhost:8000/api/Object_Enhance_Api/"
Django_Crowd_Detection_Api = "http://localhost:8000/api/Crowd_Detection_Api/"
import shutil
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



@app.route('/object_detection', methods=['GET', 'POST'])
def detection():
    print("fake video")
    if request.method == 'POST':
        print('inside post')
        if 'object_detection_file' not in request.files:
            return jsonify({"error": "No Video file provided"}), 400

        selected_checkboxes = request.form.get('selected_checkboxes')

        if selected_checkboxes:
            # Convert the selected checkboxes data from JSON string to a Python list
            import json
            selected_checkboxes = json.loads(selected_checkboxes)
            
            # Print or process the selected checkboxes
            print("Selected checkboxes (value and name):", selected_checkboxes)
            
            for checkbox in selected_checkboxes:
                print(f"Value: {checkbox['value']}, Name: {checkbox['name']}")
        else:
            return jsonify({'error': 'No checkboxes selected'}), 400

        file = request.files['object_detection_file']
        files = {'object_detection_file': (file.filename, file.stream, file.mimetype)}

        # Send the checkbox names and values as JSON (ensure this is separate from the file data)
        data = {
            'check_box_names': json.dumps([checkbox['name'] for checkbox in selected_checkboxes]),  # Send as JSON string
            'check_box_values': json.dumps([checkbox['value'] for checkbox in selected_checkboxes])  # Send as JSON string
        }

        # Send the request with both files and form-data (NOT json)
        response = requests.post(Django_Object_Detection_Api, files=files, data=data)

        try:
            return response.json(), response.status_code
        except requests.exceptions.JSONDecodeError:
            return jsonify({"error": "Invalid response from server"}), 500




@app.route('/object_enhance', methods=['GET','POST'])
def object_enhance():
    if request.method == 'POST':
        print('inside post')
        if 'object_detection_file' not in request.files:
            return jsonify({"error": "No Video file provided"}), 400

        file = request.files['object_detection_file']
        data_type = request.form.get('type')
        if data_type == "image_file":
            data_type = {"type": "image_file"} 
            files = {'object_detection_file': (file.filename, file.stream, file.mimetype)}

            response = requests.post(Django_Object_Enhance_Api, files=files, data=data_type)

            if response.status_code == 200:
                response_data = response.json()
                file_path = response_data["result"]["file_path"]
                file_name = response_data["result"]["filename"]
                static_folder = os.path.join(app.root_path, 'static', 'enhanced_images')
                os.makedirs(static_folder, exist_ok=True)  # Ensure folder exists
                
                file_path = os.path.normpath(file_path)

                # # Destination path inside Flask static folder
                static_file_path = os.path.join(static_folder, file_name)

                shutil.copy(file_path, static_file_path)

                print("filepath", file_path)
                print('static file path', static_file_path)
                
                try:
                    pass
                    os.remove(file_path)
                    print(f"Deleted original file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

                # print("file_path", file_path)
                try:
                    return jsonify({
                        "file_url": f"/static/enhanced_images/{file_name}",
                        "status": "Completed"
                    })
                    # return response.json(), response.status_code
                except requests.exceptions.JSONDecodeError:
                    return jsonify({"error": "Invalid response from server"}), 500
            
        elif data_type == "video_file":
            data_type = {"type": "video_file"} 
            files = {'object_detection_file': (file.filename, file.stream, file.mimetype)}
            response = requests.post(Django_Object_Enhance_Api, files=files, data=data_type)

            if response.status_code == 200:
                response_data = response.json()
                file_path = response_data["result"]["file_path"]
                file_name = response_data["result"]["filename"]
                static_folder = os.path.join(app.root_path, 'static', 'enhanced_videos')
                os.makedirs(static_folder, exist_ok=True)  # Ensure folder exists
                
                file_path = os.path.normpath(file_path)

                # # Destination path inside Flask static folder
                static_file_path = os.path.join(static_folder, file_name)

                shutil.copy(file_path, static_file_path)

                
                try:
                    os.remove(file_path)
                    print(f"Deleted original file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

                # print("file_path", file_path)
                try:
                    return jsonify({
                        "file_url": f"/static/enhanced_videos/{file_name}",
                        "status": "Completed"
                    })
                    # return response.json(), response.status_code
                except requests.exceptions.JSONDecodeError:
                    return jsonify({"error": "Invalid response from server"}), 500














@app.route('/Crowd_Detection', methods=['GET','POST'])
def Crowd_Detection():
    if request.method == 'POST':
        print('inside post')


        file = request.files['crowd_detection_files']
        data_type = request.form.get('type')
        if data_type == "image_file":
            data_type = {"type": "image_file"} 
            files = {'crowd_detection_file': (file.filename, file.stream, file.mimetype)}

            response = requests.post(Django_Crowd_Detection_Api, files=files, data=data_type)

            if response.status_code == 200:
                response_data = response.json()

                
                Density_map_path = response_data["result"]["Density_map"]
                Density_count = response_data["result"]["Density_count"]
                
                Density_map_name = response_data['result']['density_image_name']
                Density_count_name = response_data['result']['density_count_image_name']




                static_folder_for_map = os.path.join(app.root_path, 'static', 'map_images')
                os.makedirs(static_folder_for_map, exist_ok=True)  # Ensure folder exists
                
                static_folder_for_count = os.path.join(app.root_path, 'static', 'count_images')
                os.makedirs(static_folder_for_count, exist_ok=True)  # Ensure folder exists



                Density_map_path = os.path.normpath(Density_map_path)
                Density_count = os.path.normpath(Density_count)


                shutil.copy(Density_map_path, os.path.join(static_folder_for_map, Density_map_name))
                shutil.copy(Density_count, os.path.join(static_folder_for_count, Density_count_name))              
                
                try:
                    Density_map_path = os.path.dirname(Density_map_path)
                    Density_count = os.path.dirname(Density_count)

                    if os.path.exists(Density_map_path):
                        shutil.rmtree(Density_map_path)

                except Exception as e:
                    print(f"Error deleting file : {e}")

                # print("file_path", file_path)
                try:
                    return jsonify({
                        "Density_map_name":f"/static/map_images/{Density_map_name}",
                        "Density_count_name": f"/static/count_images/{Density_count_name}",
                        "status": "Completed"
                    })
                    # return response.json(), response.status_code
                except requests.exceptions.JSONDecodeError:
                    return jsonify({"error": "Invalid response from server"}), 500
                    # return response.json(), response.status_code

            
        elif data_type == "video_file":
            print('video file')
            data_type = {"type": "video_file"} 
            files = {'crowd_detection_file': (file.filename, file.stream, file.mimetype)}
            response = requests.post(Django_Crowd_Detection_Api, files=files, data=data_type)

            if response.status_code == 200:
                response_data = response.json()
                response_data = round(response_data['result'], 2)
                print("response_data", response_data)
                # print("file_path", file_path)
                try:
                    return jsonify({
                        "people_count":response_data,
                        "status": "Completed"
                    })
                    # return response.json(), response.status_code
                except requests.exceptions.JSONDecodeError:
                    return jsonify({"error": "Invalid response from server"}), 500



if __name__ == '__main__':
    socketio.run(app, port=8585)
