# serializers.py

from rest_framework import serializers

class ProcessAudioSerializer(serializers.Serializer):
    input_file = serializers.FileField(required=True)
    output_file = serializers.CharField(required=True)