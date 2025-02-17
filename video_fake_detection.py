import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance  # Added missing ImageEnhance
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load a pre-trained deepfake detection model
def load_detection_model(model_path="deepfake_detector_model.h5"):
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Perform Error Level Analysis (ELA)
def perform_ela(image_path, ela_quality=95):
    try:
        original = Image.open(image_path).convert('RGB')
        # Save the image at a lower quality
        compressed_path = "compressed.jpg"
        original.save(compressed_path, "JPEG", quality=ela_quality)

        compressed = Image.open(compressed_path)
        # Find the difference between the original and compressed image
        ela_image = ImageChops.difference(original, compressed)
        max_diff = max([extrema[1] for extrema in ela_image.getextrema()])

        scale = 255.0 / max_diff if max_diff > 0 else 1
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        return ela_image
    except Exception as e:
        print(f"Error during ELA: {e}")
        return None

# Extract and preprocess frames for detection
def extract_frames(video_path, frame_skip=30):
    video = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    if not video.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return frames

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if count % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # Resize for model input
            frames.append(frame)

        count += 1

    video.release()
    print(f"Extracted {len(frames)} frames from {video_path}.")
    return np.array(frames)

# Predict whether the frames are deep fakes
def predict_frames(model, frames):
    try:
        predictions = model.predict(frames / 255.0)  # Normalize frames
        fake_probabilities = predictions[:, 1]  # Assuming class index 1 is "fake"
        return fake_probabilities
    except Exception as e:
        print(f"Error during prediction: {e}")
        return np.array([])

# Visualize results
def visualize_results(frames, predictions, threshold=0.5):
    try:
        for i, frame in enumerate(frames[:10]):  # Display first 10 frames
            plt.imshow(frame)
            plt.title(f"Fake Probability: {predictions[i]:.2f}")
            plt.axis("off")
            plt.show()
    except Exception as e:
        print(f"Error during visualization: {e}")

# Main function for video analysis
def analyze_video(video_path, model_path):
    model = load_detection_model(model_path)
    if model is None:
        print("Exiting due to model loading error.")
        return

    frames = extract_frames(video_path)
    if len(frames) == 0:
        print("No frames extracted. Exiting.")
        return

    predictions = predict_frames(model, frames)
    if len(predictions) == 0:
        print("No predictions made. Exiting.")
        return

    visualize_results(frames, predictions)

    # Summarize detection results
    fake_count = np.sum(predictions > 0.5)
    real_count = len(predictions) - fake_count

    print(f"Summary: {fake_count} fake frames, {real_count} real frames.")

# Run the analysis
if __name__ == "__main__":
    video_path = r"C:\Users\Msawaiz10\Desktop\video.mp4"  # Replace with your video file path
    model_path = r"C:\Users\Msawaiz10\Downloads\Deepfake-detection-master\Deepfake-detection-master\model.h5"  # Replace with your model file path
    analyze_video(video_path, model_path)
