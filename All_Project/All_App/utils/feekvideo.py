import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance  # Added missing ImageEnhance
import time
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, "fake_video_models", "model.h5")
import os

MODEL_PATH = r"E:\P_M_services\All_Project\fake_video_models\model.h5"

if os.path.exists(MODEL_PATH):
    print("✅ Model file exists.")
else:
    print("❌ Model file NOT found.")


print("Modelapath", MODEL_PATH)

# Load a pre-trained deepfake detection model

MAX_RETRIES = 5  # Number of retries

def load_detection_model(model_path, retries=MAX_RETRIES):
    attempt = 0
    while attempt < retries:
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            model = load_model(model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            time.sleep(5)  # Wait for 5 seconds before retrying
    
    print("Max retries reached. Model could not be loaded.")
    return None


# Predict whether the frames are deep fakes
def predict_frames(model, frames):
    try:
        predictions = model.predict(frames / 255.0)  # Normalize frames
        fake_probabilities = predictions[:, 1]  # Assuming class index 1 is "fake"
        return fake_probabilities
    except Exception as e:
        print(f"Error during prediction: {e}")
        return np.array([])



