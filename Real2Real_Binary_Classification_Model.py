from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pdb
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
from datasets import Sim_Shapenet_Orientation_With_Depth
from datasets import DIVA_DoorDataset
#from ImgAug import ImageAug 
from auto_augment import AutoAugment
import tensorflow as tf 
from PIL import Image
import cv2
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize
from albumentations.pytorch import ToTensor

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, 
    IAAPiecewiseAffine, IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
import Models 

classes = ('Open', 'Closed')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #sends the process to GPU.
plt.ion()


class MaskedDepthLoss(nn.Module):
  def __init__(self,mask_val=0):
    super(MaskedDepthLoss, self).__init__()
    self.mask_val = mask_val
  # masked L1 norm
  def forward(self, depth_out, depth_gt):
    loss = torch.abs(depth_gt-depth_out)
    if self.mask_val is not None:
      mask_indices=torch.where(depth_gt == self.mask_val)
      loss[mask_indices] = 0
    return loss.mean()


def visualize_and_log_depth(out_name,model,input,current_epoch, log):
  with torch.no_grad():
    model.eval()
    fig = plt.figure()
    rgb = input-input.min()
    rgb /= rgb.max()
    rgb = rgb.cpu().transpose(0,1).transpose(1,2)
    ax = plt.subplot(1,2,1)
    ax.set_title('rgb input')
    ax.imshow(rgb)
    ax.axis('off')
    log.add_figure(out_name,fig,current_epoch,close=True)

def visualize_and_log_normal(out_name,model,input,current_epoch, log):
    with torch.no_grad():
        model.eval()
        fig = plt.figure()
        normal,_ = model.forward(torch.unsqueeze(input,0))
        rgb = input-input.min()
        rgb /= rgb.max()
        normal = normal[0]
        rgb = rgb.cpu().transpose(0,1).transpose(1,2)
        ax = plt.subplot(1,2,1)
        ax.set_title('rgb input')
        ax.imshow(rgb)
        ax.axis('off')
        ax = plt.subplot(1,2,2)
        ax.set_title('pred_normal')
        normal = normal.cpu().transpose(0,1).transpose(1,2)
        ax.imshow(normal)
        ax.axis('off')
        log.add_figure(out_name, fig, current_epoch, close=True)

def train_model(trainloader, model, criterion, criterion_l1, optimizer, epoch, writer):
    print ("Epoch: {:03d} ".format(epoch))
    print ("***************")
    model.train()
    train_conf = np.zeros((2,2))
    train_running_loss = 0.0 
    for inputs, labels in trainloader:
        images = inputs 
        #inputs = NCHW
        #have to solve the issue of converting tensor into np array for data augmentation pipeline
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        normal_pred, outputs = model(inputs)
        #pred,_,_,_ = model(inputs) 
        normal_pred = normal_pred.to(device)
        _, predicted = torch.max(outputs, 1)
        for gt, pred in zip(labels, predicted): # confusion matrix.
            train_conf[gt.item(), pred.item()] += 1
        loss = criterion(outputs, labels)
        loss.backward()
        train_running_loss += loss.item() # accumulates running loss until validation stage
        optimizer.step()
        
    trainingAcc = np.mean(train_conf.diagonal() / train_conf.sum(axis=1))
    print("Epoch: {:03d} Training-Acc: {:.04f}  Training Loss: {:.04f}\n".format(epoch, trainingAcc, train_running_loss/len(trainloader)))
    writer.add_scalar('Training Loss', train_running_loss / len(trainloader), epoch)
    writer.add_scalar('Training Acc', trainingAcc, epoch)
    return model


def test_model(testloader, model, criterion, epoch, writer):
    print("In test phase:\n")
    model.eval()

    testLoss = 0.0
    test_conf =  np.zeros((2,2))
    with torch.no_grad():
        for data in testloader:
            images, labels = data  # input for the testloader
            images = images.to(device)
            labels = labels.to(device)
            images = images.float()
            normal, outputs = model(images)
            loss = criterion(outputs, labels)
            testLoss += loss.item()
            _, predicted = torch.max(outputs, 1)  # torch.max returns the max in the output
            for gt, pred in zip(labels, predicted): # confusion matrix.
                test_conf[gt.item(), pred.item()] += 1

            # for a single batch, you read in the label
    #visualize_and_log_normal('normal_pred_real', model, images[0], epoch, writer)
    testAcc = np.mean(test_conf.diagonal() / test_conf.sum(axis=1)) 
    print("Epoch: {:03d} testLoss: {:.04f} \n".format(epoch, testLoss/ len(testloader)))

    #print("finished test data set")
    print("Total Accuracy: {:.04f} \n".format(testAcc))
    writer.add_scalar('Test Loss', testLoss/ len(testloader), epoch)
    writer.add_scalar('Test Acc', testAcc, epoch)

    return testAcc

def process_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_name", type=str,
                      default='baseline_real2real',
                      help='Name of experiment')
  parser.add_argument("--where_to_save", type=str,
                      default='/data/bo/exp_settings',
                      help='where to save run arguments')
  parser.add_argument("--lr", type=float,
                      default=0.001,
                      help='LR')
  parser.add_argument("--num_classes", type=int,
                      default=2,
                      help='num classes for classification')
  parser.add_argument("--num_epoch", type=int
                        ,default=50,
                        help='number of epochs to run')
  parser.add_argument("--auto_augment", type=bool
                        ,default=False,
                        help='whether to apply CIFAR10 extracted policies')
  parser.add_argument("--self_augment", type=bool
                        ,default=False,
                        help='whether to apply provided data augmentation pipeline')
  return vars(parser.parse_args())

def main(args):

    if not os.path.exists(os.path.join('/data/bo/TensorBoardDiva/Real2Real',args['exp_name'])):
        os.makedirs(os.path.join('/data/bo/TensorBoardDiva/Real2Real',args['exp_name']))

    writer = SummaryWriter(os.path.join('/data/bo/TensorBoardDiva/Real2Real',args['exp_name']))

    BestModelAcc = 0.0
    val_transform = [transforms.ToPILImage()]

    if args['auto_augment']:
        train_transform = [transforms.ToPILImage()]
        train_transform.append(AutoAugment())
        train_transform.extend(
            [transforms.Resize((256,256)),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        train_transform = transforms.Compose(train_transform)
    elif args['self_augment']:
        train_transform = Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
            Resize(256, 256), 
            RandomCrop(224, 224),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensor()
        ])
        
    else: 
        train_transform = [transforms.ToPILImage()]
        train_transform.extend(
            [transforms.Resize((256,256)),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        train_transform = transforms.Compose(train_transform)

    val_transform.extend(
       [transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    val_transform = transforms.Compose(val_transform)
    trainSet = DIVA_DoorDataset(real_root = '/data/diva/door_train_loso0000_v2', transforms=train_transform)
    testSet = DIVA_DoorDataset(real_root = '/data/diva/door_val_loso0000_v2',val_transforms=val_transform)

    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=4,
                                              # specification of the batch size, batches the sample
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=4,
                                              # specification of the batch size, batches the sample
                                              shuffle=True, num_workers=2)

    model = Models.ResNet101_depth_with_multi(num_classes = args['num_classes'])
    model.load_state_dict(torch.load('/data/bo/savedModel/UNetWithDepth4DoorLabelRun1'))#Change to Normal Model 
    #model.classifier = nn.Linear(2048, args['num_classes'])
    #parameters = []
    #for name, param in model.named_parameters():
        #param.requires_grad = False
        #if 'classifier' in name:
            #param.requires_grad = True
            #parameters.append(param) 

    #manipulates the last layer based on number of classes to identify. 
    model_ft = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion_l1 = MaskedDepthLoss(mask_val=0)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args['lr'], momentum=0.9) #why do you create a linear layer with num_ft.

    #step = epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.9)
    
    for epoch in range(args['num_epoch']):
        model = train_model(trainloader, model, criterion,criterion_l1, optimizer_ft, epoch, writer)
        exp_lr_scheduler.step() 
        testAcc = test_model(testloader, model, criterion, epoch, writer) 

if __name__ == '__main__':
    args = process_args()
    if not os.path.exists(os.path.join(args['where_to_save'])):
        os.makedirs(os.path.join(args['where_to_save'],'{:s}.json'.format(args['exp_name'])))
    with open(os.path.join(args['where_to_save'],'{:s}.json'.format(args['exp_name'])),'wt') as fp:
        json.dump(args, fp, indent=2)

    main(args)

