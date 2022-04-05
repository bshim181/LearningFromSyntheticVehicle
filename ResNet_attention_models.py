import torch
import torchvision
import pdb
import torch.nn.functional as F
import torch.nn as nn

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

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

class ResNet101_Attention_Model(torch.nn.Module):
    #From the input image, you make a depth prediction.
    #Then incoporate that information as you perform classification task. 
  def __init__(self, num_classes, attention=True, normalize_attn=True):
    super(ResNet101_Attention_Model, self).__init__()
    resnet = torchvision.models.resnet101(pretrained=True)
    base_model_out_size = 2048
    self.attention = attention

    self.layer0 = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    self.layer1 = torch.nn.Sequential(resnet.layer1)  # 256 1/4 size 
    self.layer2 = torch.nn.Sequential(resnet.layer2)  # 512 1/8 size 
    self.layer3 = torch.nn.Sequential(resnet.layer3)  # 1024 1/16 size 
    self.layer4 = torch.nn.Sequential(resnet.layer4)  # 2048 1/32 size 

    self.classifier = torch.nn.Linear(2048, num_classes)

    self.decoderLayer1 = torch.nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0)
    self.decoderLayer2 = torch.nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)

    if self.attention:
        self.projector1 = ProjectorBlock(256,1024)
        self.projector2 = ProjectorBlock(512,1024) 
        self.projector3 = ProjectorBlock(1024,2048)
        self.attn1 = LinearAttentionBlock(in_features= 1024, normalize_attn=normalize_attn)
        self.attn2 = LinearAttentionBlock(in_features= 1024, normalize_attn=normalize_attn)
        self.attn3 = LinearAttentionBlock(in_features= 1024, normalize_attn=normalize_attn)
        self.classifier = torch.nn.Linear(in_features = 1024, out_features = num_classes, bias=True)

  def forward(self, x):
    # Bottom-up
    c1 = self.layer0(x)
    c2 = self.layer1(c1)  # 256 channels, 1/4 size
    l1 = c2 #256 channels 56 Size
    c3 = self.layer2(c2)  # 512 channels, 1/8 size
    l2 = c3 #512 channels 28 size 
    c4 = self.layer3(c3)  # 1024 channels, 1/16 size
    l3 = c4  #1024 channel 14 size 
    c5 = self.layer4(c4)  # 2048 channels, 1/32 size

    if self.attention:
        c5 = self.decoderLayer1(c5)
        #g = self.decoderLayer2(c6)
        g = F.avg_pool2d(c5, 7)
        f1, g1 = self.attn1(self.projector1(l1), g)
        f2, g2 = self.attn2(self.projector2(l2), g)
        f3, g3 = self.attn3(l3, g) 
        #g = torch.cat((g1,g2,g3), dim=1)
        g = g1 + g2 + g3
        x = self.classifier(g) #channel size of 2048 
    else:
        g = F.avg_pool2d(c5, 7).view(-1, 2048)
        f1, f2, f3 = None, None, None 
        x = self.classifier(g)
    return [x, f1, f2, f3]
