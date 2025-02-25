


import os
import cv2
import torch
import torchvision.transforms as standard_transforms
from torch.autograd import Variable
from networks.Counter import Counter
from PIL import Image
import numpy as np
import math
import time

# Define counting model initialization and evaluation function
def initialize_counter(model_path, model_type="ADML", mode="DME", device=0):
    net = Counter(model_name=model_type, mode=mode)
    net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    net.cuda(device)
    net.eval()
    return net

def evaluate_counter(net, img_path, transform, device=0):
    img = Image.open(img_path)
    if img.mode == "L":
        img = img.convert('RGB')

    img = transform(img)
    img = img.view(1, img.size(0), img.size(1), img.size(2))
    w = math.ceil(img.shape[2] / 16) * 16
    h = math.ceil(img.shape[3] / 16) * 16
    data_list = torch.FloatTensor(1, 3, int(w), int(h)).fill_(0)
    data_list[:, :, 0:img.shape[2], 0:img.shape[3]] = img
    img = data_list

    with torch.no_grad():
        img = Variable(img).cuda(device)
        torch.cuda.synchronize()
        start_time = time.time()
        pred_map = net.test_forward(img, None)
        torch.cuda.synchronize()
        end_time = time.time()
        diff_time = end_time - start_time

    pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]
    pred = np.sum(pred_map) / 100.0
    return int(pred)

# Set parameters for counting model
arguments_strModelStateDict = 'adml_small_vehicle.pth'
arguments_strModel = "ADML"
arguments_strMode = 'DME'
arguments_intDevice = 0

# Initialize the model and transforms
mean_std = ([0.36475515365600586, 0.36875754594802856, 0.34205102920532227],
            [0.2001768797636032, 0.19185248017311096, 0.1892034411430359])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])

counter_net = initialize_counter(arguments_strModelStateDict, arguments_strModel, arguments_strMode, arguments_intDevice)

# Video processing and frame extraction with counting
video_path = r"E:\P_M_services\crowd_1\testing\vehicle.mp4"
cap = cv2.VideoCapture(video_path)

output_folder = "screenshots"
os.makedirs(output_folder, exist_ok=True)

rect_height = 100
saved_frame_counter = 0
total_vehicle_count = 0  # Initialize total vehicle count

ret, frame = cap.read()
if not ret:
    print("Error reading video.")
    cap.release()
    exit()

height, width = frame.shape[:2]
rect_y_position = height - rect_height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rect_start = (0, rect_y_position)
    rect_end = (width, rect_y_position + rect_height)
    cv2.rectangle(frame, rect_start, rect_end, (0, 0, 255), 2)

    if rect_y_position <= 0:
        frame_filename = os.path.join(output_folder, f"saved_frame_{saved_frame_counter}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")

        # Evaluate the saved frame
        count = evaluate_counter(counter_net, frame_filename, img_transform, arguments_intDevice)
        print(f"Frame {saved_frame_counter} Count: {count}")

        # Update total vehicle count
        total_vehicle_count += count

        saved_frame_counter += 1
        rect_y_position = height - rect_height

    rect_y_position -= 10

    cv2.imshow("Frame with Rectangle", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Display the total vehicle count
print(f"Total Vehicle Count: {total_vehicle_count}")
