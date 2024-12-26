from torchvision.models import vit_b_16
import torch
from torch import nn
import math
import torch.nn.functional as F
import torchvision.ops as ops

class ConvSamePadding(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride):
        super(ConvSamePadding, self).__init__()
        self.input_size = input_size
        padding = self.calculate_same_padding(input_size, kernel_size, stride)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def calculate_same_padding(self, input_size, kernel_size, stride): 
        return math.ceil((stride * (input_size - 1) + kernel_size - input_size) / 2)
    
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == self.input_size
        return self.conv(x)
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(3, 8, 3, stride=3, padding=1)
        self.se1 = ops.SqueezeExcitation(8, 2)
        self.cnn2 = nn.Conv2d(8, 16, 3, stride=3, padding=1)
        self.se2 = ops.SqueezeExcitation(16, 4)
        self.cnn3 = nn.Conv2d(16, 32, 3, stride=3, padding=1)
        self.se3 = ops.SqueezeExcitation(32, 8)
        self.cnn4 = nn.Conv2d(32, 64, 3, stride=3, padding=1)
        self.se4 = ops.SqueezeExcitation(64, 16)
        self.cnn5 = nn.Conv2d(64, 128, 3, stride=3, padding=1)
        self.se5 = ops.SqueezeExcitation(128, 32)
        # self.cnn1 = ConvSamePadding(input_size=224, in_channels=3, out_channels=8, kernel_size=3, stride=3)
        # self.cnn2 = ConvSamePadding(input_size=74, in_channels=8, out_channels=16, kernel_size=3, stride=3)
        # self.cnn3 = ConvSamePadding(input_size=24, in_channels=16, out_channels=32, kernel_size=3, stride=3)
        # self.cnn4 = ConvSamePadding(input_size=8, in_channels=32, out_channels=64, kernel_size=3, stride=3)
        # self.cnn5 = ConvSamePadding(input_size=2, in_channels=64, out_channels=128, kernel_size=3, stride=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.se1(x)
        x = self.relu(x)
        # x = F.max_pool2d(x, kernel_size=3)
        x = self.cnn2(x)
        x = self.se2(x)
        x = self.relu(x)
        # x = F.max_pool2d(x, kernel_size=3)
        x = self.cnn3(x)
        x = self.se3(x)
        x = self.relu(x)
        # x = F.max_pool2d(x, kernel_size=3)
        x = self.cnn4(x)
        x = self.se4(x)
        x = self.relu(x)
        # x = F.max_pool2d(x, kernel_size=3)
        x = self.cnn5(x)
        x = self.se5(x)
        x = self.relu(x)
        # x = F.max_pool2d(x, kernel_size=2)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vmodel = vit_b_16(pretrained=False)
        self.vmodel.encoder.layers = self.vmodel.encoder.layers[0:5]
        self.vmodel.heads.head = nn.Identity()
        self.cnn_model = CNN()
        self.fc = nn.Linear(896, 3)

    def forward(self, x):
        x1 = self.vmodel(x)
        x2 = self.cnn_model(x)
        # print(x1.shape, x2.shape)
        x2 = x2.flatten(1)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x)
        return x