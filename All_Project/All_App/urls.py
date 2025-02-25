# users/urls.py
from django.urls import path
from .views import *

urlpatterns = [
    # path('Upload_Data/', RagData.as_view(), name='Upload_Data'),
    # path('Response_output/',RagData_Generation.as_view(),name='Response_output'),
    # path('Video_Summarization/', Video_Summarization.as_view(), name='Video_Summarization'),
    # path('Text_Translation/', Text_Translation.as_view(), name='Text_Translation'),
    # path('Audio_Transcription/', Audio_Transcription.as_view(),name='Audio_Transcription'),
    path('Deep_Video_Detection/', Deep_Video_Detection.as_view(), name='Deep_Video_Detection'),
    path('Video_Converter/', Video_Converter_Api.as_view(), name='Video_Converter'),
    path('Video_Compresser/', Video_Compresser.as_view(), name='Video_Compresser'),
    path('Object_Detection_Api/', Object_Detection_Api.as_view(), name='Object_Detection_Api'),
    path('Object_Enhance_Api/', Object_Enhance_Api.as_view(), name='Object_Enhance_Api'),
    path('Crowd_Detection_Api/', Crowd_Detection_Api.as_view(), name='Crowd_Detection_Api')
]
