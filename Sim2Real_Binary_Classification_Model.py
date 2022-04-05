
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
import Models
import Models_N1
from auto_augment import AutoAugment
import albumentations as A
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine, IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
from albumentations import CenterCrop, Normalize, HorizontalFlip, Resize, RandomCrop
from albumentations.pytorch import ToTensorV2

writer = None 
#plt.ion()
classes = ('Open', 'Closed')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #sends the process to GPU.

def visualize_and_log_depth(out_name,model,input,current_epoch, log):
  with torch.no_grad():
    model.eval()
    fig = plt.figure()
    depth,_ = model.forward(torch.unsqueeze(input,0))
    rgb = input-input.min()
    rgb /= rgb.max()
    rgb = rgb.cpu().transpose(0,1).transpose(1,2)
    depth_vis = (depth[0,0]*255).cpu().numpy().astype(np.uint8)
    ax = plt.subplot(1,2,1)
    ax.set_title('rgb input')
    ax.imshow(rgb)
    ax.axis('off')
    ax = plt.subplot(1, 2, 2)
    ax.set_title('pred depth')
    ax.imshow(depth_vis)
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

def validate_model(validloader, model, batch_count, epoch, criterion, writer):
    model.eval()
    confusion = np.zeros((2, 2))
    val_running_loss = 0.0
    for valImage, valLabels in validloader:
        with torch.no_grad():
            valImage = valImage.to(device)
            valLabels = valLabels.to(device)
            outputs = model(valImage)
            loss = criterion(outputs[1], valLabels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs[1], 1)
            for gt, pred in zip(valLabels, predicted):
                confusion[gt.item(), pred.item()] += 1

    tempModelAcc = np.mean(confusion.diagonal() / confusion.sum(axis=1))
    print("BatchCount: {:03d} Epoch: {:03d}  Validation-Acc: {:.04f}  ValidationLoss: {:.04f}\n".format(batch_count, epoch, tempModelAcc, val_running_loss/len(validloader)))

    log_epoch = epoch + 1
     # ...log the running loss
    writer.add_scalar('Validation Loss',
                val_running_loss / len(validloader),
                log_epoch * batch_count)
    writer.add_scalar('Validation Acc', tempModelAcc, log_epoch*batch_count)
    #if batch_count % 1000 == 0 or batch_count == 0:
        #visualize_and_log_normal('normal_pred_real', model, valImage[0], epoch, writer) 
    return tempModelAcc


def train_model(model, criterion,criterion_l1, optimizer, scheduler, trainloader, validloader, writer, num_epochs =2):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    tempModelAcc = 0
    BestModelAcc = 0
    criterion = criterion.to(device)
    criterion_l1 = criterion_l1.to(device)

    model.train();

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        batch_count = 0
        train_conf = np.zeros((2,2)) # confusion matrix -> holds the accuracy
        train_running_loss = 0.0
        normal_running_loss = 0.0 
        for inputs, depth, normal, orientation, elevation, labels in trainloader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            depth = depth.to(device)
            normal = normal.to(device)
            if batch_count % 100 == 0 and batch_count != 0:
                #now in validation phase 
                trainingAcc = np.mean(train_conf.diagonal() / train_conf.sum(axis=1))
                print("BatchCount: {:03d} Epoch: {:03d} Training-Acc: {:.04f}  Training-loss: {:.04f}  normal-loss: {:.04f}\n".format(batch_count, epoch, trainingAcc, train_running_loss/100.0, normal_running_loss/100.0))
                #if batch_count % 1000 == 0 or batch_count == 0:
                    #visualize_and_log_normal('normal_pred_Sim', model, inputs[0], epoch, writer)
                #visualize_and_log_depth('depth_pred_Sim', model, inputs[0], epoch, writer)
                tempModelAcc = validate_model(validloader, model, batch_count, epoch, criterion, writer)
                
                log_epoch = epoch + 1
                writer.add_scalar('Training Loss', train_running_loss / 100.0,
                        log_epoch * batch_count)
                writer.add_scalar('Training Acc', trainingAcc, log_epoch * batch_count)
                writer.add_scalar('normal_loss', normal_running_loss / 100.0, log_epoch * batch_count) 
                
                normal_running_loss = 0.0
                train_running_loss = 0.0
                train_conf = np.zeros((2,2))
                
            if tempModelAcc > BestModelAcc:
                best_model_wts = copy.deepcopy(model.state_dict())
                BestModelAcc = tempModelAcc

            model.train()
            batch_count += 1
            # forward
            #obtain the depth prediction and the output prediction from the input 
            normal_pred, outputs = model(inputs)         
            _, predicted = torch.max(outputs, 1)
            #compute depth loss. 
            for gt, pred in zip(labels, predicted): # confusion matrix.
                train_conf[gt.item(), pred.item()] += 1
           
            #normal_loss = criterion_l1(normal_pred, normal)
            loss_cls = criterion(outputs, labels)
            #Computes the loss based on depth pred and image prediction
            #loss = 0.5*loss_cls + 0.5*normal_loss 
            loss = loss_cls
            #Update weights based on loss calcualted from depth and cls 
            loss.backward()
            train_running_loss += loss # accumulates running loss until validation stage
            #normal_running_loss += normal_loss
            optimizer.step()
            scheduler.step()
        torch.save(model.state_dict(),'/data/bo/savedModel/UNetN1WithoutNormal4DoorLabelRun1')
    model.load_state_dict(best_model_wts)
    return model;

def test_model(model, criterion, testloader):
    torch.save(model.state_dict(),'/data/bo/savedModel/UNetN1WithoutNormal4DoorLabelRun1')
    print("In test phase:\n")
    model.eval()

    testLoss = 0.0
    batch_count = 0
    test_conf = np.zeros((2,2))
    with torch.no_grad():
        for data in testloader:
            images, labels = data  # input for the testloader
            images = images.to(device)
            labels = labels.to(device)
            images = images.float()
            outputs = model(images)
            loss = criterion(outputs[1], labels)
            testLoss += loss.item()
            if (batch_count % 100 and batch_count != 0):
                print("Batch: {:03d} testLoss: {:.04f}".format(batch_count, testLoss/100))
                testLoss = 0.0
            _, predicted = torch.max(outputs[1], 1)  # torch.max returns the max in the output
            batch_count += 1
            for gt, pred in zip(labels, predicted): # confusion matrix.
                test_conf[gt.item(), pred.item()] += 1
    testAcc = np.mean(test_conf.diagonal() / test_conf.sum(axis=1))
    print("finished test data set")
    print("Total Accuracy: {:.04f}".format(testAcc))    

def process_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_name", type=str,
                      default= 'sim2real',
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
  parser.add_argument("--self_augment", type=bool,
                      default=False,
                      help='whether to apply augmentation pipeline or not')
  parser.add_argument("--auto_augment", type=bool,
                      default=False,
                      help='whether to apply extracted augmentation policies')
  return vars(parser.parse_args())


def main(args):

    if not os.path.exists(os.path.join('/data/bo/TensorBoardDiva/Sim2Real',args['exp_name'])):
        os.makedirs(os.path.join('/data/bo/TensorBoardDiva/Sim2Real',args['exp_name']))

    writer = SummaryWriter(os.path.join('/data/bo/TensorBoardDiva/Sim2Real',args['exp_name']))


    if args['self_augment']:
        train_transform = Compose([
            A.HorizontalFlip(p=1),
            A.IAAAdditiveGaussianNoise(p=1),
            A.Blur(blur_limit=11, p=1),
            A.RandomBrightness(p=1),
            A.CLAHE(p=1),
            A.Resize(224,224),
            A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        depth_transform = Compose([
            A.HorizontalFlip(p=1),
            A.IAAAdditiveGaussianNoise(p=1),
            A.Blur(blur_limit=11, p=1),
            A.RandomBrightness(p=1),
            A.CLAHE(p=1),
            A.Resize(56,56),
        ])

    train_transform = transforms.Compose([ 
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    depth_transform = transforms.Compose([
        transforms.Resize((56,56)),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    #initializes data sets
    trainSet = Sim_Shapenet_Orientation_With_Depth(transforms=train_transform, depth_transforms = depth_transform, include_normal = 1, auto_aug = args['auto_augment'])
    validSet = DIVA_DoorDataset(real_root = '/data/diva/door_train_loso0000_v2', transforms=val_transform)
    testSet = DIVA_DoorDataset(real_root = '/data/diva/door_val_loso0000_v2',transforms=val_transform)


    #Creates an iterable of each data set.
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size= 8,
                                              shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validSet, batch_size=4,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=4,
                                              shuffle=True, num_workers=2)

    model = Models_N1.ResNet101_depth_with_multi(num_classes = args['num_classes'])
    model.load_state_dict(torch.load('/data/bo/savedModel/UNetN1WithoutNormal4DoorLabelRun1'))
    model_ft = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    criterion_l1 = MaskedDepthLoss(mask_val=0)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args['lr'], momentum=0.9) #why do you create a linear layer with num_ft.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10000, gamma=0.9)

    model = train_model(model, criterion,criterion_l1, optimizer_ft, exp_lr_scheduler, trainloader, validloader, writer)
    test_model(model, criterion, testloader)



if __name__ == '__main__':
    args = process_args()
    if not os.path.exists(os.path.join(args['where_to_save'])):
        os.makedirs(os.path.join(args['where_to_save'],'{:s}.json'.format(args['exp_name'])))
    with open(os.path.join(args['where_to_save'],'{:s}.json'.format(args['exp_name'])),'wt') as fp:
        json.dump(args, fp, indent=2)

    main(args)
