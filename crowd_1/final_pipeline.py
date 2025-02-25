import cv2
import numpy as np
import torch
from torch.autograd import Variable
from modeling import *
import os
from matplotlib import cm as CM
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
# Set paths for the pretrained model and the video
MODEL_PATHS = {
    'sha': {
        'MARNet': r"E:\crowd_1\x_net.pth"
    }
}
video_path = r"E:\P_M_services\crowd_1\testing\vehicle.mp4"
output_folder = "screenshots"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

def preprocess_image(cv2im):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])  # Convert BGR to RGB
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to CHW
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    im_as_ten = torch.from_numpy(im_as_arr).float().unsqueeze(0)  # Add batch dimension
    return Variable(im_as_ten, requires_grad=True)

def load_model(model_name, model_paths, dataset='sha'):
    if model_name == 'MARNet':
        return MARNet(load_model=model_paths[dataset][model_name], downsample=1, objective='dmp+amp')

def generate_density_map(model, frame, divide=50):
    img = preprocess_image(frame)
    if torch.cuda.is_available():
        img = img.cuda()
    outputs = model(img)
    dmp = outputs[0].squeeze().detach().cpu().numpy() if torch.cuda.is_available() else outputs[0].squeeze().detach().numpy()
    dmp = dmp / divide
    return dmp, dmp.sum()

def process_video():
    # Load the model
    model = load_model('MARNet', MODEL_PATHS, dataset='sha')
    if torch.cuda.is_available():
        model = model.cuda()

    # Set up video capture
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error reading video.")
        cap.release()
        return

    height, width = frame.shape[:2]
    rect_height = 100  # Height of the rectangle
    rect_y_position = height - rect_height  # Start from the bottom
    saved_frame_counter = 0
    total_people_count = 0  # Initialize total people count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Define the rectangle position
        rect_start = (0, rect_y_position)
        rect_end = (width, rect_y_position + rect_height)

        # Draw the rectangle on the current frame
        cv2.rectangle(frame, rect_start, rect_end, (0, 0, 255), 2)

        # Check if the top of the rectangle reaches the top of the frame
        if rect_y_position <= 0:
            # Generate density map and estimate people count for the current frame
            dmp, frame_people_count = generate_density_map(model, frame)
            total_people_count += frame_people_count

            # Save the frame with the rectangle and density map
            frame_filename = os.path.join(output_folder, f"saved_frame_{saved_frame_counter}.jpg")
            cv2.imwrite(frame_filename, frame)
            # print(f"Saved {frame_filename} - Estimated People Count: {frame_people_count}")
            saved_frame_counter += 1

            # Save the density map as a heatmap image
            density_map_filename = os.path.join(output_folder, f"density_map_{saved_frame_counter}.png")
            plt.imsave(density_map_filename, dmp, cmap='hot')

            # Reset the rectangle position to continue tracking
            rect_y_position = height - rect_height

        # Move the rectangle up for the next frame
        rect_y_position -=4  # Adjust the speed of movement here if necessary (1 to 10)

        # Optional: Display the frame with the rectangle for preview
        cv2.imshow("Frame with Rectangle", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Print total people count across all frames
    print(f"Total Estimated People Count Across All Frames: {round(total_people_count)}")

# Run the video processing
if __name__ == '__main__':
    process_video()
