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

parser = argparse.ArgumentParser(description='train model')
parser.add_argument('--data_dir', type=str, action="store", default="flowers", help='data location')
parser.add_argument('--gpu', dest ='gpu', action='store', default="gpu", help='choose GPU if available')
parser.add_argument('--save_dir',type=str, dest="save_dir", action="store", default='chechpoint.pth', help='save location for checkpoint')
parser.add_argument('--arch',dest='arch', action="store", default="vgg16", type=str, help='choose pretrained - vgg16')
parser.add_argument('--learning_rate',type=float, dest="learning_rate", action="store", default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=25088, help='number of hidden units')
parser.add_argument('--epochs',type=int, dest="epochs", action="store", default=2, help='number of epochs')
parser.add_argument('--dropout', type=float, dest="dropout", action="store", default=0.5, help='dropout')

args=parser.parse_args()

path=args.data_dir
gpu = args.gpu
save_dir=args.save_dir
model_arch = args.arch
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


def load_data(train_dir, valid_dir):
    
    
    data_transforms_train=transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    data_transforms_valid = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224,0.225])])

    image_datasets_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)
    
    trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=32)
    
    return trainloader, validloader

from collections import OrderedDict    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1000)), 
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1000, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc3', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))]))
    
model = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(model)

def train_model():
    
    trainloader, validloader = load_data(train_dir, valid_dir)
    for epoch in range(epochs):
        print('epochs', epoch)
        tn_loss=0
        v_loss=0
        batches=0
        accuracy=0
        for batch in trainloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred=model.forward(images)
            loss = loss_f(y_pred, labels)
            tn_loss+=loss.item()
            loss.backward()
            optimizer.step()
        print("loss_of_training=", tn_loss/len(trainloader))
        
        with torch.no_grad():
            model.eval()
            for batch in validloader:
                v_images, v_labels = batch
                v_images, v_labels = v_images.to(device),v_labels.to(device)
                
                y_pred=model(v_images)
                loss=loss_f(y_pred, v_labels)
                v_loss+=loss.item()
                
                ps=torch.exp(model(v_images))
                top_p, top_class=ps.topk(1)
                equal=(top_class==v_labels.reshape(*top_class.shape))
                accuracy +=torch.mean(equal.type(torch.FloatTensor))
            model.train()
            print("valid_loss=", v_loss/len(validloader))
            print('v_accuracy', accuracy/len(validloader))
            

def save_model():
    checkpoint = {'model_arch' : model_arch,
                  'hidden_units' : hidden_units,
                  'learning_rate' : lr,
                  'epochs' : epochs,
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict,
                  'state_dict': model.state_dict()
                  }
    torch.save(checkpoint, 'checkpoint.pth');

if __name__== "__main__":
    print('training start')
    load_data(train_dir, valid_dir)
    print(torch.cuda())
    train_model()


            