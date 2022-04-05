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
from datasets import DIVA_DoorDataset_MultiLabel
import Models
from auto_augment import AutoAugment
import albumentations as A
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine, IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
from albumentations import CenterCrop, Normalize, HorizontalFlip, Resize, RandomCrop
from albumentations.pytorch import ToTensorV2
from attention_models import  AttnVGG_before
from attention_utilities import visualize_attn_softmax1
from attention_utilities import visualize_attn_sigmoid
from ResNet_attention_models import ResNet101_Attention_Model
import Models_N1

writer = None
classes = ('Open', 'Closed')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #sends the process to GPU.

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


def validate_model(validloader, model, batch_count, epoch, criterion, writer):
    model.eval()
    val_running_loss = 0.0
    recall = 0.0
    precision = 0.0

    for data in validloader:
        with torch.no_grad():
            valImage, valLabels = data
            valImage = valImage.to(device)
            valLabels = torch.stack(valLabels[0])
            valLabels = valLabels.float()
            valLabels = valLabels.to(device) # labels = [FL, FR, BL, BR, T], Values of 0 and 1
            valLabels = valLabels.transpose(0,1)
            normal,outputs = model(valImage)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, valLabels)
            val_running_loss += loss.item()

            #Calculated for every 8 batches 
            alpha = 0.000001
            preds = outputs>0.5
            num_recalled = preds[torch.where(valLabels==1)].sum()
            recall = recall + num_recalled/(len(torch.where(valLabels==1)[0])+alpha)

            #Precision -> how many of my predictions have matched to the gt
            num_precision = len(torch.where(preds==1)[0])
            precision += valLabels[torch.where(preds==1)].sum()/(num_precision+alpha)

    #f1 score = harmonic avg
    recall, precision = recall/len(validloader), precision/len(validloader)
    f1 = 2 / (1/recall + 1/precision)

    print("BatchCount: {:03d} Epoch: {:03d}  V-Precision: {:.04f}  V-Recall: {:.04f}  V-loss: {:.04f}  V-f1: {:.04f}\n".format(batch_count, epoch, precision, recall, val_running_loss/len(validloader), f1))

    log_epoch = epoch + 1
    # ...log the running loss
    writer.add_scalar('Validation Loss',
                val_running_loss / len(validloader),
                log_epoch * batch_count)
    writer.add_scalar('Validation Recall', recall, log_epoch*batch_count)
    writer.add_scalar('Validation Precision', precision, log_epoch*batch_count)
    writer.add_scalar('Validation f1', f1, log_epoch*batch_count)
    
    if(normal != None):
        if batch_count % 1000 == 0 or batch_count == 100:
            visualize_and_log_normal('Normal_Pred_Real', model, valImage[0], epoch, writer)
    
    return f1

def train_model(model, criterion, criterion_l1, optimizer, scheduler, trainloader, validloader, writer, num_epochs = 5):
    best_model_wts = copy.deepcopy(model.state_dict())
    tempModelf1 = 0
    BestModelf1 = 0
    recall = 0.0 
    precision = 0.0 
    normal_running_loss = 0.0 
    criterion = criterion.to(device)

    model.train()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        batch_count = 0
        train_running_loss = 0.0
        for inputs, depth, normal, orientation, elevation, labels in trainloader:
            inputs = inputs.to(device)
            normal = normal.to(device)
            optimizer.zero_grad()
            if batch_count % 100 == 0 and batch_count != 0: #now in validation phase
                #f1 score = harmonic avg
                #Precision Recur
                recall, precision = recall/100, precision/100
                f1 = 2 / (1/recall + 1/precision) 
                print("BatchCount: {:03d} Epoch: {:03d} T-Recall: {:.04f}  T-Precision: {:.04f} T-f1: {:.04f} T-Loss: {:.04f}  Normal_Loss: {:.04f}\n".format(batch_count, epoch, recall, precision, f1, train_running_loss/100, normal_running_loss/100))
                tempModelf1 = validate_model(validloader, model, batch_count, epoch, criterion, writer)

                log_epoch = epoch + 1
                writer.add_scalar('Training Loss', train_running_loss/100,
                        log_epoch * batch_count)
                writer.add_scalar('Training Recall', recall,
                        log_epoch * batch_count)
                writer.add_scalar('Training Precision', precision, log_epoch * batch_count)
                writer.add_scalar('Training F1', f1, log_epoch * batch_count)

                train_running_loss = 0.0
                normal_running_loss =0.0
            if tempModelf1 > BestModelf1:
                best_model_wts = copy.deepcopy(model.state_dict())
                BestModelf1 = tempModelf1
            model.train()
            batch_count += 1
            # forward
            #obtain the output prediction from the input
            #Attention is accounted for when making the output prediction 
            normal_pred,output = model(inputs)
            output = output.float()
            labels = torch.stack(labels[0])
            labels = labels.float()
            labels = labels.to(device)
            labels = labels.transpose(0,1)
            output = torch.sigmoid(output)
            loss_cls = criterion(output, labels)
            loss = loss_cls
            if (normal_pred != None):
                normal_loss = criterion_l1(normal_pred, normal)
                loss = 0.5*loss_cls + 0.5*normal_loss

            train_running_loss = loss_cls + train_running_loss
            #normal_running_loss = normal_loss + normal_running_loss 
            loss.backward()
            optimizer.step()
            scheduler.step()

            #Precision Recall

            #Recall
            alpha = 0.000001 
            preds = output>0.5
            num_recalled = preds[torch.where(labels==1)].sum()
            recall += num_recalled/(len(torch.where(labels==1)[0])+alpha)

            #Precision -> how many of my predictions have matched to the gt
            num_precision = len(torch.where(preds==1)[0])
            precision += labels[torch.where(preds==1)].sum()/(num_precision+ alpha)

            if(normal_pred != None):
                if batch_count % 1000 == 0 or batch_count == 0:
                    visualize_and_log_normal('Normal_Pred_Sim', model, inputs[0], epoch, writer)

        torch.save(model.state_dict(),'/data/bo/attention_model/Sim2RealUNetN1WithoutNormalMultiLabelClassificationRun1')
    model.load_state_dict(best_model_wts)
    return model;

def test_model(model, criterion, testloader):
    torch.save(model.state_dict(),'/data/bo/attention_model/Sim2RealUNetN1WithoutNormalMultiLabelClassificationRun1')
    print("In test phase:\n")
    model.eval()
    recall, precision = 0.0, 0.0
    testLoss = 0.0
    batch_count = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data  # input for the testloader
            images = images.to(device)
            labels = torch.stack(labels[0])
            labels = labels.float()
            labels = labels.to(device) # labels = [FL, FR, BL, BR, T], Values of 0 and 1 
            labels = labels.transpose(0,1) 
            normal,pred = model(images)
            outputs = torch.sigmoid(pred)
            loss = criterion(outputs, labels)
            testLoss += loss.item()
            #calculating Precision and Recall
            alpha = 0.000001
            preds = outputs>0.5
            num_recalled = preds[torch.where(labels==1)].sum()
            recall = recall + num_recalled/(len(torch.where(labels==1)[0])+alpha)

            #Precision -> how many of my predictions have matched to the gt
            num_precision = len(torch.where(preds==1)[0])
            precision += labels[torch.where(preds==1)].sum()/(num_precision+ alpha)

    precision, recall = precision/len(testloader), recall/len(testloader)
    f1 = 2 / (1/recall + 1/precision)
    print("Test-F1: {:.04f}  Test Precision: {:.04f}  Test Recall: {:.04f}  Test Loss: {:.04f}\n".format(f1, precision, recall, testLoss/len(testloader)))
    
    #logging to tensorboard
    writer.add_scalar('Test Loss', testLoss/len(testloader),
                        batch_count)
    writer.add_scalar('Test Recall', recall,
                        batch_count)
    writer.add_scalar('Test Precision', precision, batch_count)
    writer.add_scalar('Test F1', f1, batch_count)
    
    print("finished test data set")

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
        #Do I have to center crop the images first? 
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    depth_transform = transforms.Compose([
        transforms.Resize((56,56)),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #initializes data sets
    trainSet = Sim_Shapenet_Orientation_With_Depth(transforms=train_transform, depth_transforms = depth_transform, include_normal = 1, auto_aug = args['auto_augment'])
    validSet = DIVA_DoorDataset_MultiLabel(real_root = '/data/diva/door_train_loso0000_v2', transforms=val_transform)
    testSet = DIVA_DoorDataset_MultiLabel(real_root = '/data/diva/door_val_loso0000_v2',transforms=val_transform)


    #Creates an iterable of each data set.
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size= 8,
                                              shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validSet, batch_size=8,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=8,
                                              shuffle=True, num_workers=2)

    model = Models_N1.ResNet101_depth_with_multi(num_classes = 5)
    model_ft = model.to(device)
    criterion = nn.BCELoss()
    criterion_l1 = criterion_l1 = MaskedDepthLoss(mask_val=0)
    criterion = criterion.to(device)
    criterion_l1 = criterion_l1.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args['lr'], momentum=0.9) #why do you create a linear layer with num_ft.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10000, gamma=0.9)

    model = train_model(model, criterion, criterion_l1, optimizer_ft, exp_lr_scheduler, trainloader, validloader, writer)
    test_model(model, criterion, testloader)



if __name__ == '__main__':
    args = process_args()
    if not os.path.exists(os.path.join(args['where_to_save'])):
        os.makedirs(os.path.join(args['where_to_save'],'{:s}.json'.format(args['exp_name'])))
    with open(os.path.join(args['where_to_save'],'{:s}.json'.format(args['exp_name'])),'wt') as fp:
        json.dump(args, fp, indent=2)

    main(args)

