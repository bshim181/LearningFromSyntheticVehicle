import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import torch
import pdb
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pickle
import glob
from sklearn.metrics import pairwise_distances_argmin

import csv
import torchvision
from ImgAug import ImageAug
from auto_augment import AutoAugment
from torchvision import transforms 
import random

operations = {
    'ShearX': lambda img, magnitude: shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: translate_y(img, magnitude),
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: AutoAugment.auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: AutoAugment.invert(img, magnitude),
    'Equalize': lambda img, magnitude: AutoAugment.equalize(img, magnitude),
    'Solarize': lambda img, magnitude: AutoAugment.solarize(img, magnitude),
    'Posterize': lambda img, magnitude: AutoAugment.posterize(img, magnitude),
    'Contrast': lambda img, magnitude: AutoAugment.contrast(img, magnitude),
    'Color': lambda img, magnitude: AutoAugment.color(img, magnitude),
    'Brightness': lambda img, magnitude: AutoAugment.brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: AutoAugment.sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: AutoAugment.cutout(img, magnitude),
}

class DIVA_DoorDataset_MultiLabel(Dataset):
     #Iterator initialization
    #Iterates over the dataset.
    #will data augment only in the train dataset.
  def __init__(self,
               real_root='/data/diva/door_train_loso0000_v2',
               name=False,
               transforms=None,
               val_transforms=None,
               ):
    self.real_root = real_root
    self.indices = []
    self.transform = transforms
    self.val_transform = val_transforms
    self.name = name
    self._init()
  def __len__(self):
    return len(self.indices)

  def __getitem__(self, idx):
    paths, labels = self.indices[idx] #holds each data samples. 
    crop = cv2.imread(paths) #loads an image from the file given at specified path.  
    #pdb.set_trace()
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    if self.transform:
        crop = self.transform(crop)
    if self.val_transform:
        crop = self.val_transform(crop)

    if self.name:
      return crop, labels, paths
    else:
      return crop, labels
  def _init(self):
    import csv 
    if 'train' in self.real_root:
        Open_csvfile = open('/data/diva/v2_opened_train_files.csv', newline='')
        Closed_csvfile = open('/data/diva/v2_closed_train_files.csv', newline='')
    else:
        Open_csvfile = open('/data/diva/v2_opened_val_files.csv', newline='')
        Closed_csvfile = open('/data/diva/v2_closed_val_files.csv', newline='')
    labelReader = csv.reader(Open_csvfile, delimiter=',')
    for sample_name, row in zip(sorted(os.listdir(os.path.join(self.real_root,'Opened'))), labelReader):
        #FL, FR, BL, BR, T = row[1], row[2], row[3], row[4], row[5] 
        im_path = os.path.join(self.real_root, 'Opened',sample_name)
        doorState = []
        doorState.append((int(row[1]),int(row[2]), int(row[3]), int(row[4]), int(row[5])))
        self.indices.append((im_path, doorState))
    
    labelReader = csv.reader(Closed_csvfile, delimiter=',')
      #append 1 or 0 depending on the state of doors in indices. 
    for sample_name, row in zip(sorted(os.listdir(os.path.join(self.real_root, 'Closed'))),labelReader):
        im_path = os.path.join(self.real_root,'Closed', sample_name)
        doorState = [] 
        doorState.append((int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])))
        self.indices.append((im_path, doorState))



class DIVA_DoorDataset(Dataset):
    #Iterator initialization
    #Iterates over the dataset. 
    #will data augment only in the train dataset. 
  def __init__(self,
               real_root='/data/diva/door_train_loso0000',
               name=False,
               transforms=None,
               val_transforms=None,
               ):
    self.real_root = real_root
    self.indices = [] 
    self.transform = transforms
    self.val_transform = val_transforms 
    self.name = name
    self._init()
  def __len__(self):
    return len(self.indices)

  def __getitem__(self, idx):
    paths, labels = self.indices[idx] #holds each data samples. 
    crop = cv2.imread(paths) #loads an image from the file given at specified path.  
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    if self.transform:
        crop = self.transform(crop)
    if self.val_transform:
        crop = self.val_transform(crop)
    if self.name:
      return crop, labels, paths
    else:
      return crop, labels

    
  def _init(self):
        #labelReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #for row in labelReader:
            #row[1]=FL, row[2]=FR, row[3]=BL, row[4]=BR, row[5]=Trunk
    for sample_name in os.listdir(os.path.join(self.real_root,'Opened')):
      im_path = os.path.join(self.real_root, 'Opened',sample_name)
      #sample_name
      self.indices.append((im_path, 1))
      #append 1 or 0 depending on the state of doors in indices. 
    for sample_name in os.listdir(os.path.join(self.real_root, 'Closed')):
      im_path = os.path.join(self.real_root,'Closed', sample_name)
      self.indices.append((im_path, 0))

def apply_augmentations_depth(depth,pol1, mag1, pol2, mag2, pol_range, range_num):
    if range_num < pol_range[1]:
        depth = pol1(depth, pol_range[2])
    if range_num < pol_range[4]:
        depth = pol2(depth, pol_range[5])
    return depth

def apply_augmentations_normal(normal,pol1, mag1, pol2, mag2, pol_range, range_num):
    if range_num < pol_range[1]:
        normal = pol1(normal, pol_range[2])
    if range_num < pol_range[4]:
        normal = pol2(normal, pol_range[5])
    return normal

#Simulation dataset initialization. 
class Sim_Shapenet_Orientation_With_Depth(Dataset):
  def __init__(self,
               sim_roots=['/data/ue_data/0415_vehicle_parts_with_depth_part_seg_normal'],
               size_degrade=1,
               include_normal=0,
               meshes_to_include=[],
               transforms=None,
               depth_transforms=None,
               auto_aug = False
               ):
    self.roots = sim_roots
    self.include_normal = include_normal
    self.size_degrade = size_degrade
    self.transforms = transforms
    self.meshes_to_include = meshes_to_include
    #meshes is like a generated key
    self.data = []
    self.depth_transforms = depth_transforms
    self.auto_augment = auto_aug 
    self.randomSeedCounter = 0
    self._init()

  def __len__(self):
    return len(self.data)


#single iteration of simulated data. 
  def __getitem__(self, i):

    if self.include_normal:
        img_path, seg_path,depth_path,normal_path, orientation, elevation, trunkState = self.data[i]
    else:
        img_path, seg_path,depth_path, orientation, elevation, trunkState = self.data[i]

    #contains 4 different types of informations : depth, normal, rgb, seg 
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg = cv2.imread(seg_path)

    #incorporating seg values into the image input
    x1, y1, x2, y2 = self.seg_to_box(seg)
    img = img[y1:y2, x1:x2, :]
    

    depth = cv2.imread(depth_path)[:,:,0]
    tensorTransform = transforms.Compose([transforms.ToTensor()])

    ToPIL = transforms.ToPILImage()
    aug = AutoAugment()
    img = ToPIL(img)
    depth = ToPIL(depth)

    img = aug(img)
    pol1, mag1, pol2, mag2, pol_range, range_num = AutoAugment.return_applied_policy(aug)  
    #have to fix -> augment 
    if self.auto_augment and self.include_normal != 1:
        depth = apply_augmentations_depth(depth,pol1, mag1, pol2, mag2, pol_range, range_num)

    img = self.transforms(img) 
    depth = self.depth_transforms(depth)
    depth /= depth.max()
    if self.include_normal:
        normal = cv2.imread(normal_path)
        normal = normal[y1:y2,x1:x2,:]
        normal = ToPIL(normal)
        if self.auto_augment:
            normal = apply_augmentations_normal(normal, pol1, mag1, pol2, mag2, pol_range, range_num)
        normal = self.depth_transforms(normal)
        return img, depth, normal, orientation,elevation, trunkState
    else:
        return img, depth, orientation, elevation, trunkState 


  def seg_to_box(self, seg):
    inds1, inds2 = np.where(seg[:, :, 0] == 0)
    y1 = min(inds1)
    y2 = max(inds1)
    x1 = min(inds2)
    x2 = max(inds2)

    dx = x2-x1
    dy = y2-y1
    x1 = int(max(0,x1 - dx*0.1))
    y1 = int(max(0,y1 - dy*0.1))
    x2 = int(min(seg.shape[1],x2 + dx*0.1))
    y2 = int(min(seg.shape[0],y2 + dy*0.1))

    return x1, y1, x2, y2

  def _init(self):
    print('setting up ShapenetOrientation with Depth')
    for root in self.roots:
      meta_files = glob.glob(os.path.join(root, 'meta', '*.json'))

      for metafile in meta_files:
        inst_name = metafile.split('/')[-1][:8]
        for car_state in ['Default_Closed','Default_Opened']:
          rgb_path = os.path.join(root, car_state, 'rgb', '{:06d}.png'.format(int(inst_name)))
          #use RGB this week / Image Augmentation

          seg_path = os.path.join(root, car_state, 'seg', '{:06d}.png'.format(int(inst_name)))
          depth_path = os.path.join(root, car_state, 'depth', '{:06d}.png'.format(int(inst_name)))
          if self.include_normal:
            normal_path = os.path.join(root, car_state, 'normal', '{:06d}.png'.format(int(inst_name)))

          if not os.path.exists(rgb_path):
            continue
          meta = json.load(open(metafile, 'r'))
          if True:
#          if meta['mesh'] in self.meshes_to_include:
            orientation = meta['az'] # if the trunk is open, set the state to i1
            elevation = meta['el']
            distance = meta['dist']
            #azmuth, elevation, distance from the point of view)
            trunkOpen = meta['angles']
            trunkState = 0
            if trunkOpen[0] > 5 or trunkOpen[1] > 5 or trunkOpen[2] > 5 or trunkOpen[3] > 5 :
               trunkState =1
            #trunkState = []
            #FL, FR, RL, RR, Trunk = 0, 0, 0, 0, 0 
            #if trunkOpen[0] > 5:  Front Left Door
                #FL = 1 
            #if trunkOpen[1] > 5: Front Right Door
                #FR =1 
            #if trunkOpen[2] > 5: Rear Left Door
                #RL =1 
            #if trunkOpen[3] > 5: Rear Right Door.
                #RR =1 
            #if trunkOpen[5]> 5: Trunk open state
                #Trunk = 1 
            #trunkState.append((FL, FR, RL, RR, Trunk))

            """
            An orientation array that holds the angle of the door located at each position 

            Front left
            Front right
            Rear L
            Rear R
            Hood
            Trunk
            """

            if self.include_normal:  
                self.data.append((rgb_path, seg_path,depth_path,normal_path, orientation, elevation, trunkState))
            else:
                self.data.append((rgb_path, seg_path,depth_path, orientation, elevation, trunkState))
    print('Done.. ')
    print("Total of {:d} instances ".format(len(self.data)))


if __name__ == '__main__' :
  dataset = Sim_Shapenet_Orientation_With_Depth(
               sim_roots=['/data/ue_data/0415_vehicle_parts_with_depth_part_seg_normal'],
               size_degrade=1,
               include_normal=0,
               meshes_to_include=[],
               transforms=None
               )

 
