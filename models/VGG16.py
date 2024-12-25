"""
    This file contains the code for implementation of 'Research on Classification of Clothing Fabrics Images Based on Convolutional Neural Network'
    http://dx.doi.org/10.1007/978-3-319-99695-0_12
"""
import torch
from torchvision import models

class VGG16(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Sequential(
            torch.nn.Linear(num_features, 3),
            torch.nn.Softmax()
        )

    def forward(self, input):
        outputs = self.model(input)
        return outputs