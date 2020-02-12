import numpy as np
import os, random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import argparse


p = argparse.ArgumentParser(description='prediction')
p.add_argument('-c',action='checkpoint',help = 'path to checkpoint', default='/checkpoint.pth')
p.add_argument('in_img',action='store',type=str, help='data location', default='flowers')
p.add_argument('-s',action='store',type=str,help='save location for checkpoint', default='chechpoint.pth')
p.add_argument('-g','--gpu',type=str,help='choose GPU if available',default='cpu')
p.add_argument('-k', '--topk', type=int, default=5)

args_pred=p.parse_args()
img_dir=p.in_img
output = p.k
process = p.g
path = p.s






device = torch.device('cuda' if gpu_mode=='gpu' and torch.cuda.is_available() else 'cpu')

def load_model(file_path):
    checkpoint = torch.load(file_path)
    in_size = checkpoint['input_size']
    out_size = checkpoint['output_size']
    arch = checkpoint['model']
    model.load_state_dict=checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    model = load_checkpoint('checkpoint.pth')
    
    return model


def process_image(image):
    
    img = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
    image = transform(img)
    
    return image

def predict(image_path, model, topk=5):
    
    model.cuda()
    model.eval()
    img = process_image(image_path)
    img = torch.from_numpy(img).float().cuda()
    img = torch.unsqueeze(img, dim=0)
    output = model.forward(img)
    pred = torch.exp(output).topk(topk)
    probs = pred[0][0].cpu().data.numpy()
    classes = pred[1][0].cpu().data.numpy()
    labels = [idx_to_class[i] for i in classes]
    
    return probs, labels




