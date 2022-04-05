import torch
import torchvision
import pdb
import torch.nn.functional as F


class ResNet101_depth_with_multi(torch.nn.Module):
    #From the input image, you make a depth prediction.
    #Then incoporate that information as you perform classification task. 
  def __init__(self, num_classes, out_channels=1, out_channels_normal=3):
    super(ResNet101_depth_with_multi, self).__init__()
    resnet = torchvision.models.resnet101(pretrained=True)
    base_model_out_size = 2048

    self.layer0 = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    self.layer1 = torch.nn.Sequential(resnet.layer1)  # 256
    self.layer2 = torch.nn.Sequential(resnet.layer2)  # 512
    self.layer3 = torch.nn.Sequential(resnet.layer3)  # 1024
    self.layer4 = torch.nn.Sequential(resnet.layer4)  # 2048

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

    # Depth prediction
    self.predict1 = torch.nn.Sequential(
        torch.nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
    )

    self.predict2 = torch.nn.Sequential(
        torch.nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
    )
    #Normal Prediction
    self.predict_normal1 = torch.nn.Sequential(
        torch.nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
    )
    self.predict_normal2 = torch.nn.Sequential(
        torch.nn.Conv2d(64, out_channels_normal, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
    )
    self.encoder2 = torch.nn.Sequential(
        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, stride =2),
        torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, stride =2),
        torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(1024),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, stride =2)

    )
    self.classifier = torch.nn.Linear(1024, num_classes)
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

    _, _, H, W = x.size()  # batchsize N,channel,height,width

    # Bottom-up
    c1 = self.layer0(x)
    c2 = self.layer1(c1)  # 256 channels, 1/4 size
    c3 = self.layer2(c2)  # 512 channels, 1/8 size
    c4 = self.layer3(c3)  # 1024 channels, 1/16 size
    c5 = self.layer4(c4)  # 2048 channels, 1/32 size

    
    #flat_multi_preds,recons_preds
    #Why is it refer to as a lateral eature map?

    # Top-down
    p5 = self.toplayer(c5) #input 2048 channels
    p4 = self._upsample_add(p5, self.latlayer1(c4))  # 256 channels, 1/16 size (1028 -> 64) 
    p4 = self.smooth1(p4) #what is smoothing layer for? 
    p3 = self._upsample_add(p4, self.latlayer2(c3))  # 256 channels, 1/8 size (512 -> 64)
    p3 = self.smooth2(p3)  # 256 channels, 1/8 size
    p2 = self._upsample_add(p3, self.latlayer3(c2))  # 256, 1/4 size (256 -> 64) 
    p2 = self.smooth3(p2)  # 256 channels, 1/4 size

    #Second Encoder Layer

    depth = self.predict2(self.predict1(p2)) #prediction output depth map : size of 64 channels.
    #Making classification predictions based on the depth feature map produced from p2.
    e1 = self.encoder2(p2) #expansive
    features = F.avg_pool2d(e1,7).view(-1,1024) #extracting feature map. 
    output = self.classifier(features)

    return depth, output

