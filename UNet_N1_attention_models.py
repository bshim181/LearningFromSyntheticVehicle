import torch
import torchvision
import pdb
import torch.nn.functional as F
import torch.nn as nn
from UNet_attention_models import UNet_with_attention

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

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class UNet_N1_with_attention(torch.nn.Module):
    #From the input image, you make a depth prediction.
    #Then incoporate that information as you perform classification task. 
  def __init__(self, num_classes, attention=False, normalize_attn=False):
    super(UNet_with_attention, self).__init__()
    resnet = torchvision.models.resnet101(pretrained=True)
    base_model_out_size = 2048
    self.attention = attention

    self.layer0 = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    self.layer1 = torch.nn.Sequential(resnet.layer1)  # 256
    self.layer2 = torch.nn.Sequential(resnet.layer2)  # 512
    self.layer3 = torch.nn.Sequential(resnet.layer3)  # 1024
    self.layer4 = torch.nn.Sequential(resnet.layer4)  # 2048

    """

#Normal Prediction
    self.predict_normal1 = torch.nn.Sequential(
        torch.nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
    )
    self.predict_normal2 = torch.nn.Sequential(
        torch.nn.Conv2d(64, out_channels_normal, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
    )

    """


    # Top layer
    self.toplayer = torch.nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

    # Lateral layers decoding in lateral layer
    self.latlayer1 = torch.nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
    self.latlayer2 = torch.nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
    self.latlayer3 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    # Smooth layers -> gets bigger depth features by increasing kernel sizes. 
    self.smooth1 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.smooth2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.smooth3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    self.classifier = torch.nn.Linear(2048, num_classes)
    self.normal_classifier = torch.nn.Linear(1024, numclasses)

    if self.attention:
        self.projector = ProjectorBlock(256,1024) 
        self.attn1 = LinearAttentionBlock(in_features= 1024, normalize_attn=normalize_attn)
        self.attn2 = LinearAttentionBlock(in_features= 1024, normalize_attn=normalize_attn)
        self.attn3 = LinearAttentionBlock(in_features= 1024, normalize_attn=normalize_attn)
        self.classifier = torch.nn.Linear(2048, num_classes)

  def _upsample_add(self, x, y):
    '''Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.

      (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    e map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    '''
    #output size is equal to the lateral feature map.
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners = True) + y

  def forward(self, x): 
      # Bottom-up
    c1 = self.layer0(x)
    c2 = self.layer1(c1)  # 256 channels, 1/4 size
    c3 = self.layer2(c2)  # 512 channels, 1/8 size
    c4 = self.layer3(c3)  # 1024 channels, 1/16 size
    c5 = self.layer4(c4)  # 2048 channels, 1/32 size
    
    # Top-down
    p5 = self.toplayer(c5) #input 2048 channels, output 256 channels 
    p4 = self._upsample_add(p5, self.latlayer1(c4))  # 256 channels, 1/16 size (1028 -> 64)
    p4 = self.smooth1(p4) #what is smoothing layer for?
    l1 = p4 #256
    p3 = self._upsample_add(p4, self.latlayer2(c3))  # 256 channels, 1/8 size (512 -> 64)
    p3 = self.smooth2(p3)  # 256 channels, 1/8 size
    l2 = p3 #256
    p2 = self._upsample_add(p3, self.latlayer3(c2))  # 256, 1/4 size (256 -> 64)
    p2 = self.smooth3(p2)  # 256 channels, 1/4 size 
    l3 = p2 
    f1,f2,f3 = None, None, None
    if self.attention:
        g = F.avg_pool2d(p2,7) 
        f1, g1 = self.attn1(self.projector(l1),g)
        f2, g2 = self.attn2(self.projector(l2),g)
        f3, g3 = self.attn3(self.projector(l3),g)
        g = g1 + g2 + g3  #Uses attention features at different levels in the model.
        normal = self.normal_classifier(g)
        g = F.avg_pool2d(c5,7).view(-1,2048) #extracting feature map.
        output = self.classifier(g)
        #normal prediction here 
    else:
        #Classfication from Layer C5.
        normal = None
        g = F.avg_pool2d(c5,7).view(-1,2048) #extracting feature map.
        output = self.classifier(g)

    return [output, normal, f1, f2, f3]
