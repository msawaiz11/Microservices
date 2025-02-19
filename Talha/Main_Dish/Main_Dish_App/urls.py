from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('process-audio/', views.process_audio, name='process_audio'),
]