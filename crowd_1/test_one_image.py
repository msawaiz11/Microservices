


import cv2
import numpy as np
import torch
from torch.autograd import Variable
from modeling import *
import os
from matplotlib import cm as CM
import matplotlib.pyplot as plt

model_paths = {
    'sha': {
        'MARNet': r"E:\crowd_1\x_net.pth",
    }
}

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

def load_model(models, model_paths, dataset='sha'):
    pretrained_models = {}
    for model in models:
        if model == 'MARNet':
            pretrained_model = MARNet(load_model=model_paths[dataset][model], downsample=1, objective='dmp+amp')
        pretrained_models[model] = pretrained_model
    return pretrained_models

def img_test(pretrained_model, img_path, divide=50, ds=1):
    img = cv2.imread(img_path)
    img = preprocess_image(img)
    if torch.cuda.is_available():
        img = img.cuda()
    outputs = pretrained_model(img)
    if torch.cuda.is_available():
        dmp = outputs[0].squeeze().detach().cpu().numpy()
    else:
        dmp = outputs[0].squeeze().detach().numpy()
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

    # Display the image with overlay
    # cv2.imshow("Image with Estimated Count", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    # Load only MARNet for the sha dataset
    model = load_model(['MARNet'], model_paths, dataset='sha')

    # Move model to GPU if available
    if torch.cuda.is_available():
        model['MARNet'] = model['MARNet'].cuda()

        
    img_path = r"E:\P_M_services\crowd_1\testing\150.jpg"

    # Generate output file names based on the input file name
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    density_map_path = f"{base_name}_density_map.png"
    output_image_path = f"{base_name}_output_with_count.jpg"

    # Run the test function with the model and image path
    dmp, total_count = img_test(model['MARNet'], img_path, divide=50, ds=1)

    # Save the density map
    plt.imsave(density_map_path, dmp, cmap='hot')
    print(f"Density map saved to {density_map_path}")

    # Overlay the count on the original image, save, and display it
    overlay_count_on_image(img_path, total_count, output_image_path)
