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
from UNet_N1_attention_models import UNet_N1_with_attention

writer = None
#plt.ion()
classes = ('Open', 'Closed')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #sends the process to GPU.
import torchvision.utils as utils

def visualize_attention_train(Image,img, predicted,writer, epoch, model):
    I_train = utils.make_grid(img, nrow=1, normalize=True, scale_each=True)
    writer.add_image('Sim/image', I_train, epoch)
    vis_fun = visualize_attn_softmax1
    min_up_factor = 1
    with torch.no_grad():
        __, c1,c2,c3 = model(Image)
        attn1 = vis_fun(img, I_train, c1, up_factor=min_up_factor, nrow=1)
        writer.add_image('Sim/attention_map_1', attn1, epoch)
        attn2 = vis_fun(img, I_train, c2, up_factor=min_up_factor*2, nrow=1)
        writer.add_image('Sim/attention_map_2 ', attn2, epoch)
        attn3 = vis_fun(img, I_train, c3, up_factor=min_up_factor*4, nrow=1)
        writer.add_image('Sim/attention_map_3', attn3, epoch)


def visualize_attention_validation(Image, img, predicted, writer, epoch, model):
    I_train = utils.make_grid(img, nrow=1, normalize=True, scale_each=True)
    writer.add_image('val/image', I_train, epoch)
    vis_fun = visualize_attn_softmax1
    min_up_factor = 1 
    with torch.no_grad():
        __, c1,c2,c3 = model(Image) 
        attn1 = vis_fun(img, I_train, c1, up_factor=min_up_factor, nrow=1)
        writer.add_image('val/attention_map_1 ', attn1, epoch)
        attn2 = vis_fun(img, I_train, c2, up_factor=min_up_factor*2, nrow=1)
        writer.add_image('val/attention_map_2', attn2, epoch)
        attn3 = vis_fun(img, I_train, c3, up_factor=min_up_factor*4, nrow=1)
        writer.add_image('val/attention_map_3', attn3, epoch)


def visualize_attention_test(img, writer, model, epoch=1):
    I_train = utils.make_grid(img, nrow=1, normalize=True, scale_each=True)
    writer.add_image('test/image', I_train, epoch)
    vis_fun = visualize_attn_softmax1
    min_up_factor = 1
    with torch.no_grad():
        __, c1,c2,c3 = model(img)
        attn1 = vis_fun(I_train, c1, up_factor=min_up_factor, nrow=1)
        writer.add_image('test/attention_map_1', attn1, epoch)
        attn2 = vis_fun(I_train, c2, up_factor=min_up_factor*2, nrow=1)
        writer.add_image('test/attention_map_2', attn2, epoch)
        attn3 = vis_fun(I_train, c3, up_factor=min_up_factor*4, nrow=1)
        writer.add_image('test/attention_map_3', attn3, epoch)

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
    for data in validloader:
        with torch.no_grad():
            valImage, valLabels = data
            valImage, valLabels = valImage.to(device), valLabels.to(device)
            outputs,__,__,__ = model(valImage)
            loss = criterion(outputs, valLabels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
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
    if batch_count % 1000 == 0 or batch_count == 100:
        visualize_attention_validation(valImage,valImage[0], predicted, writer, epoch, model)
    # have to visualize attention maps 
    # overlap attention map onto the image. 
    return tempModelAcc

def train_model(model, criterion, criterion_l1, optimizer, scheduler, trainloader, validloader, writer, num_epochs = 5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    tempModelAcc = 0
    BestModelAcc = 0
    criterion = criterion.to(device)

    model.train()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        batch_count = 0
        train_conf = np.zeros((2,2)) # confusion matrix -> holds the accuracy
        train_running_loss = 0.0
        for inputs, depth, normal, orientation, elevation, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device) 
            normal = normal.to(device) 
            optimizer.zero_grad()
            if batch_count % 100 == 0 and batch_count != 0:
                #now in validation phase
                pdb.set_trace()
                trainingAcc = np.mean(train_conf.diagonal() / train_conf.sum(axis=1))
                print("BatchCount: {:03d} Epoch: {:03d} Training-Acc: {:.04f}  Training-loss: {:.04f}\n".format(batch_count, epoch, trainingAcc, train_running_loss/100.0))
                tempModelAcc = validate_model(validloader, model, batch_count, epoch, criterion, writer)

                log_epoch = epoch + 1
                writer.add_scalar('Training Loss', train_running_loss / 100.0,
                        log_epoch * batch_count)
                writer.add_scalar('Training Acc', trainingAcc, log_epoch * batch_count)
                
                train_running_loss = 0.0
                train_conf = np.zeros((2,2))

            if tempModelAcc > BestModelAcc:
                best_model_wts = copy.deepcopy(model.state_dict())
                BestModelAcc = tempModelAcc
            model.train() 
            batch_count += 1
            # forward
            #obtain the output prediction from the input
            #Attention is accounted for when making the output prediction 
            output,__,__,__ = model(inputs)
            _, predicted = torch.max(output, 1)
            #compute loss.
            for gt, pred in zip(labels, predicted): # confusion matrix.
                train_conf[gt.item(), pred.item()] += 1
            loss = criterion(output, labels)
            train_running_loss = loss + train_running_loss 
            loss.backward()
            optimizer.step()
            scheduler.step()
            if batch_count % 1000 == 0:
                visualize_attention_train(inputs, inputs[0], predicted, writer, epoch, model)  

    model.load_state_dict(best_model_wts)
    return model;

def test_model(model, criterion, testloader):
    torch.save(model.state_dict(),'/data/bo/attention_model/Sim2RealResNetAttentionModelFiltOrient4DoorLabelRun1')
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
            pred,__,__,__ = model(images)
            loss = criterion(pred, labels)
            testLoss += loss.item()
            _, predicted = torch.max(pred, 1)  # torch.max returns the max in the output
            # for a single batch, you read in the label
            for gt, pred in zip(labels, predicted): # confusion matrix.
                test_conf[gt.item(), pred.item()] += 1

            # for a single batch, you read in the label

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
        #transforms.Resize((32,32)),
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
        #transforms.Resize((32,32)),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #initializes data sets
    trainSet = Sim_Shapenet_Orientation_With_Depth(transforms=train_transform, depth_transforms = depth_transform, include_normal = 1, auto_aug = args['auto_augment'])
    validSet = DIVA_DoorDataset(real_root = '/data/diva/door_train_loso0000_v2', transforms=val_transform)
    testSet = DIVA_DoorDataset(real_root = '/data/diva/door_val_loso0000_v2',transforms=val_transform)


    #Creates an iterable of each data set.
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size= 8,
                                              shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validSet, batch_size=8,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=8,
                                              shuffle=True, num_workers=2)

    model = ResNet101_Attention_Model(num_classes = 2) 
    model_ft = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    criterion_l1 = MaskedDepthLoss(mask_val=0)
    #criterion_l1 = nn.L1Loss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args['lr'], momentum=0.9) #why do you create a linear layer with num_ft.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5000, gamma=0.9)

    model = train_model(model, criterion, criterion_l1, optimizer_ft, exp_lr_scheduler, trainloader, validloader, writer)
    test_model(model, criterion, testloader)



if __name__ == '__main__':
    args = process_args()
    if not os.path.exists(os.path.join(args['where_to_save'])):
        os.makedirs(os.path.join(args['where_to_save'],'{:s}.json'.format(args['exp_name'])))
    with open(os.path.join(args['where_to_save'],'{:s}.json'.format(args['exp_name'])),'wt') as fp:
        json.dump(args, fp, indent=2)

    main(args)

