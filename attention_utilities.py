import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
import torchvision 
from torchvision import transforms
import pdb

def visualize_attn_softmax1(image, I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    N,C,W,H = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    a = cv2.resize(a[0,0].cpu().numpy(),(224,224))
    a = torch.tensor(a)
    a = a.view(1,1,224,224)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.5 * img + 0.5 * attn #overlaying attention heat map on top of the image.
    return torch.from_numpy(vis).permute(2,0,1)

def visualize_attn_softmax(image, I, a, up_factor, nrow):
    # imag
    vis_img = image[0]-image[0].min()
    vis_img = vis_img/image[0].max()
    vis_img *= 255
    vis_img = vis_img.transpose(0,1).transpose(1,2).cpu().numpy()

    N,C,W,H = a.size()
    atn = F.softmax(a.view(N,C,-1), dim=2).view(N,C,W,H)
    if up_factor > 1:
        atn = F.interpolate(atn, scale_factor=up_factor, mode='bilinear', align_corners=False)

    attn_val = (atn[0,0].cpu()*255).numpy().astype(np.uint8)
    atn_vis = cv2.resize(attn_val, (224,224))
    atn_vis = np.stack((np.zeros((224,224)),np.zeros((224,224)),atn_vis))
    atn_vis = atn_vis.transpose(1,2,0)
    atn_vis = cv2.applyColorMap(atn_vis.astype(np.uint8), cv2.COLORMAP_JET)
    atn_vis = cv2.cvtColor(atn_vis, cv2.COLOR_BGR2RGB)
    vis = cv2.addWeighted(atn_vis.astype(np.uint8),0.4,vis_img.astype(np.uint8),0.6,0)
    return torch.from_numpy(vis).permute(2,0,1)

def visualize_attn_sigmoid(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    a = torch.sigmoid(c)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=False)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)
