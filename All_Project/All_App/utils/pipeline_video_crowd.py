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
       
        'MARNet': r"E:\P_M_services\All_Project\weights\x_net.pth"
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

def load_model_video(model_name, model_paths, dataset='sha'):
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

