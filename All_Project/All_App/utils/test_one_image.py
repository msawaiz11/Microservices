


import cv2
import numpy as np
import torch
from torch.autograd import Variable
# from modeling import *
from modeling import *
import os
from matplotlib import cm as CM
import matplotlib.pyplot as plt
# import sys
# sys.path.append(r"e:\MARUNet\modeling")

# Set your pretrained model path here
model_paths = {
    'sha': {
        'MARNet': r" All_Project\weights\x_net.pth"
    }
}



def preprocess_image(cv2im, target_size=(512, 512)):  # Resize to 512x512
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    cv2im = cv2.resize(cv2im, target_size)  # Resize image
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])  # Convert BGR to RGB
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert to CHW format
    
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    
    im_as_ten = torch.from_numpy(im_as_arr).float().unsqueeze(0)  # Add batch dimension
    return Variable(im_as_ten, requires_grad=False)  # No need for gradient computation


def load_model(models, model_paths, dataset='sha'):
    pretrained_models = {}
    for model in models:
        if model == 'MARNet':
            pretrained_model = MARNet(load_model=model_paths[dataset][model], downsample=1, objective='dmp+amp')
            pretrained_model.eval()  # Set to evaluation mode
            pretrained_model.to(device)  # Move to selected device
        pretrained_models[model] = pretrained_model
    return pretrained_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def img_test(pretrained_model, img_path, divide=50, ds=1):
    img = cv2.imread(img_path)
    img = preprocess_image(img)
    
    # Move to selected device (GPU if available, otherwise CPU)
    img = img.to(device)
    pretrained_model = pretrained_model.to(device)

    with torch.no_grad():  # Disable gradient calculations to save memory
        outputs = pretrained_model(img)

    dmp = outputs[0].squeeze().cpu().numpy()  # Ensure data is moved to CPU before using numpy
    dmp = dmp / divide
    total_count = dmp.sum()
    print('Estimated people count:', total_count)
    return dmp, total_count


def overlay_count_on_image(img_path, count, output_path):
    img = cv2.imread(img_path)
    text = f"Estimated Count: {int(count)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5  # Increased font scale for larger text
    thickness = 3  # Increased thickness for bolder text
    color = (0, 0, 255)  # Red color
    shadow_color = (0, 0, 0)  # Black color for shadow
    position = (50, 100)  # Adjusted position for larger text

    # Create a shadow effect by placing the text in black slightly offset
    cv2.putText(img, text, (position[0] + 2, position[1] + 2), font, font_scale, shadow_color, thickness + 2)

    # Place the main text in the chosen color on top
    cv2.putText(img, text, position, font, font_scale, color, thickness)
    
    # Save the image with overlaid text
    cv2.imwrite(output_path, img)
    print(f"Image with count saved to {output_path}")
    return output_path
