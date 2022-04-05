import torch
import torchvision
import pdb
import torch.nn as nn
import torch.nn.functional as F
from initialize import *

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        #From Local Features l and Global Feature g, we produce compatbility score 
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH producing compatibility score c from addition of l and g. 
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H) #Softmax into range (0,1) to get attention weights
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l) # multiplies two tensor to combine  
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

'''
attention before max-pooling
'''

class AttnVGG_before(nn.Module):
    def __init__(self, im_size, num_classes, attention=False, normalize_attn=False, init='xavierUniform'):
        super(AttnVGG_before, self).__init__()
        self.attention = attention
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        #kernel_size should be dependent on the im_size 
        #aiming to produce 1x1 
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size/32), padding=0, bias=True)
        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)

            #Testing out how the size of output channels for the projector block affects performance.
            #Optimization.

            #self.projector1 = ProjectorBlock(256, 2048)
            #self.projector2 = ProjectorBlock(512, 2048)
            #self.projector3 = ProjectorBlock(1024, 2048)

            #self.attn1 = LinearAttentionBlock(in_features=2048, normalize_attn=normalize_attn)
            #self.attn2 = LinearAttentionBlock(in_features=2048, normalize_attn=normalize_attn)
            #self.attn3 = LinearAttentionBlock(in_features=2048, normalize_attn=normalize_attn)
        # final classification layer

        if self.attention:
            self.classify = nn.Linear(in_features=512*3, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")
    def forward(self, x):
        # feed forward
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        l1 = self.conv_block3(x) # /1 Size 256 
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0) # /2
        l2 = self.conv_block4(x) # /2 Size 64
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0) # /4
        l3 = self.conv_block5(x) # /4 Size 16 
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0) # /8
        x = self.conv_block6(x) # /32
        g = self.dense(x) # batch_sizex512x1x1
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC #concatenated attention layer.
            # classification layer
            x = self.classify(g) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
        return [x, c1, c2, c3]
