#Version 1 
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
from ResNet_attention_models import ResNet101_Attention_Model
from UNet_attention_models import UNet_with_attention
from attention_utilities import visualize_attn_softmax
from attention_utilities import visualize_attn_softmax1
from attention_utilities import visualize_attn_sigmoid
import torchvision.utils as utils
from UNet_attention_models import UNet_with_attention 
from datasets import DIVA_DoorDataset_MultiLabel

classes = ('Open', 'Closed')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #sends the process to GPU.
plt.ion()

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

def visualize_ResNetattention_validation(Images, img, writer, epoch, model):
    I_train = utils.make_grid(img, nrow=1, normalize=True, scale_each=True)
    writer.add_image('val/image', I_train, epoch)
    vis_fun = visualize_attn_softmax1
    min_up_factor = 1
    with torch.no_grad():
        __,c1,c2,c3 = model(Images)
        attn1 = vis_fun(img, I_train, c1, up_factor=min_up_factor, nrow=1)
        writer.add_image('val/attention_map_1 ', attn1, epoch)
        attn2 = vis_fun(img, I_train, c2, up_factor=min_up_factor*2, nrow=1)
        writer.add_image('val/attention_map_2 ', attn2, epoch)
        attn3 = vis_fun(img, I_train, c3, up_factor=min_up_factor*4, nrow=1)
        writer.add_image('val/attention_map_3 ', attn3, epoch)

def visualize_UNetattention_validation(img, writer, epoch, model):
    I_train = utils.make_grid(img, nrow=1, normalize=True, scale_each=True)
    writer.add_image('validation/image', I_train, epoch)
    vis_fun = visualize_attn_softmax1
    min_up_factor = 1
    with torch.no_grad():
        __,c1,c2,c3,c4 = model(img)
        attn1 = vis_fun(img, I_train, c1, up_factor=min_up_factor, nrow=1)
        writer.add_image('val/attention_map_1', attn1, epoch)
        attn2 = vis_fun(img, I_train, c2, up_factor=min_up_factor*2, nrow=1)
        writer.add_image('val/attention_map_2', attn2, epoch)
        attn3 = vis_fun(img, I_train, c3, up_factor=min_up_factor*4, nrow=1)
        writer.add_image('val/attention_map_3', attn3, epoch)
        attn4 = vis_fun(img, I_train, c4, up_factor=min_up_factor*4, nrow=1)
        writer.add_image('val/attention_map_4', attn4, epoch)


def train_model(trainloader, model, criterion, optimizer, epoch, writer):
    print ("Epoch: {:03d} ".format(epoch))
    print ("***************")
    model.train()
    train_running_loss = 0.0
    recall, precision, f1 = 0.0, 0.0, 0.0 
    for data in trainloader:
        images, labels = data 
        images = images.to(device)
        labels = torch.stack(labels[0])
        labels = labels.float()
        labels = labels.to(device) # labels = [FL, FR, BL, BR, T], Values of 0 and 1 
        labels = labels.transpose(0,1)
        #pred,__,__,__ = model(images)
        normal, pred = model(images)
        outputs = torch.sigmoid(pred)
        loss = criterion(outputs, labels)
        loss.backward()
        train_running_loss += loss.item()
        optimizer.step()	
        alpha = 0.000001
        preds = outputs>0.5
        num_recalled = preds[torch.where(labels==1)].sum()
        recall = recall + num_recalled/(len(torch.where(labels==1)[0])+alpha)
        num_precision = len(torch.where(preds==1)[0])
        precision += labels[torch.where(preds==1)].sum()/(num_precision+ alpha)
        

    visualize_and_log_normal('normal_pred_real', model, images[0], epoch, writer)
    epoch = epoch + 1
    precision, recall = precision/len(trainloader), recall/len(trainloader)
    f1 = 2 / (1/recall + 1/precision)
    print("Train-F1: {:.04f}  Train Precision: {:.04f}  Train Recall: {:.04f}  Train Loss: {:.04f}\n".format(f1, precision, recall, train_running_loss/len(trainloader)))
    writer.add_scalar('Train Loss', train_running_loss/len(trainloader),
                                    epoch)
    writer.add_scalar('Train Recall', recall, epoch)
    writer.add_scalar('Train Precision', precision, epoch)
    writer.add_scalar('Train F1', f1, epoch)
 
    return model


def test_model(testloader, model, criterion, epoch, writer):
    print("In test phase:\n")
    model.eval()
    recall, precision, f1 = 0.0, 0.0, 0.0
    testLoss = 0.0
    IC = 0
    with torch.no_grad():
        for data in testloader:
            IC = IC + 1
            images, labels = data  # input for the testloader
            images = images.to(device)
            labels = torch.stack(labels[0])
            labels = labels.float()
            labels = labels.to(device) # labels = [FL, FR, BL, BR, T], Values of 0 and 1 
            labels = labels.transpose(0,1)
            #pred,__,__,__= model(images)
            normal, pred = model(images)
            outputs = torch.sigmoid(pred)
            loss = criterion(outputs, labels)
            testLoss += loss.item()
            alpha = 0.000001
            preds = outputs>0.5
            num_recalled = preds[torch.where(labels==1)].sum()
            recall = recall + num_recalled/(len(torch.where(labels==1)[0])+alpha)
            num_precision = len(torch.where(preds==1)[0])
            precision += labels[torch.where(preds==1)].sum()/(num_precision+ alpha)
    
    recall, precision = recall/len(testloader), precision/len(testloader)
    f1 = 2 / (1/recall + 1/precision)
    epoch = epoch + 1
    print("Test-F1: {:.04f}  Test Precision: {:.04f}  Test Recall: {:.04f}  Test Loss: {:.04f}\n".format(f1, precision, recall, testLoss/len(testloader)))
    writer.add_scalar('Test Loss', testLoss/len(testloader),
                                    epoch)
    writer.add_scalar('Test Recall', recall, epoch)
    writer.add_scalar('Test Precision', precision, epoch)
    writer.add_scalar('Test F1', f1, epoch)

    #if epoch % 1 == 0: 
       #visualize_ResNetattention_validation(images, images[0], writer, epoch, model)
       #visualize_UNetattention_validation(images, writer, epoch, model)

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

    trainSet = DIVA_DoorDataset_MultiLabel(real_root = '/data/diva/door_train_loso0000_v2', transforms=train_transform)
    testSet = DIVA_DoorDataset_MultiLabel(real_root = '/data/diva/door_val_loso0000_v2',val_transforms=val_transform)

    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=4,
                                              # specification of the batch size, batches the sample
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=4,
                                              # specification of the batch size, batches the sample
                                              shuffle=True, num_workers=2)

    import Models_N1
    model = Models.ResNet101_depth_with_multi(num_classes = 5)
    model.load_state_dict(torch.load('/data/bo/attention_model/Sim2RealUNetWithNormalMultiLabelClassificationRun1'))
    #model.load_state_dict(torch.load('/data/bo/attention_model/Sim2RealUNetN1WithoutNormalMultiLabelClassificationRun1'))
    #model.classifier = nn.Linear(in_features=1024, out_features = 2, bias=True) 
    #parameters = []
    #for name, param in model.named_parameters():
        #param.requires_grad = False
        #if 'classifier' in name:
            #param.requires_grad = True
            #parameters.append(param) 

    model_ft = model.to(device)
    criterion = nn.BCELoss()
    crtierion = criterion.to(device)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args['lr'], momentum=0.9) #why do you create a linear layer with num_ft.

    #step = epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.9)
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(args['num_epoch']):
        model = train_model(trainloader, model, criterion, optimizer_ft, epoch, writer)
        exp_lr_scheduler.step()
        test_model(testloader, model, criterion, epoch, writer)


if __name__ == '__main__':
    args = process_args()
    if not os.path.exists(os.path.join(args['where_to_save'])):
        os.makedirs(os.path.join(args['where_to_save'],'{:s}.json'.format(args['exp_name'])))
    with open(os.path.join(args['where_to_save'],'{:s}.json'.format(args['exp_name'])),'wt') as fp:
        json.dump(args, fp, indent=2)

    main(args)

