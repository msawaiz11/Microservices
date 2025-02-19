from channels.generic.websocket import WebsocketConsumer
import json
from celery.result import AsyncResult
from Main_Dish_App.tasks import Whisper_audio_check
import os
import time
import uuid
import base64
from django.conf import settings
class TaskStatusConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass


    def receive(self, text_data):

      global status   
      data = json.loads(text_data)
      action = data.get('action')
      if data.get('type') == 'whisper_video':
          print('in whisper')
          file_data = data["file"]
          filename = file_data["name"]
          print('filename', filename)
          file_extension = os.path.splitext(filename)[1]
          filename_only = os.path.splitext(filename)[0]
          print('onlyfilename', filename_only)
          print('file_extension', file_extension)

          filename_without_ext = os.path.splitext(os.path.basename(filename))[0]
          filecontent = bytes(file_data["content"])

          file_upload_path = os.path.join(settings.BASE_DIR, "static", "uploads")
          os.makedirs(file_upload_path, exist_ok=True)  # Ensure the upload directory exists

          file_save_path = os.path.join(file_upload_path, filename)
          with open(file_save_path, 'wb') as f:
              f.write(filecontent)


          random_audio_filename = f"{uuid.uuid4()}.mp3"
          output_path = os.path.join(file_upload_path, random_audio_filename)

          print('outputpath', output_path)

    

          # Call the Celery task with the correct paths
          result = Whisper_audio_check.delay(file_save_path, output_path)


          while True:
              resultid = AsyncResult(result.id)
              status = resultid.status
              response = {
              'task_id': result.id,
              'status': status,
              'result': resultid.result if resultid.successful() else None
              }
              self.send(text_data=json.dumps(response))
              if status == 'SUCCESS':
                  break
              time.sleep(3)
      else:               
          task_id = data.get('task_id')
          if task_id:
              print('insde a async')
              result = AsyncResult(task_id)
              print('result', result.status)
              response = {                    
                  'type' : status,
                  'task_id': task_id,
                  'status': result.status,
                  'conversion_type' : conversion_type
                  # 'result': result.result
              }
              self.send(text_data=json.dumps(response))