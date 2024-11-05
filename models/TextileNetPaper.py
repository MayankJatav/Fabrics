"""
    This file contains the code for implementation of 'TextileNet: A Deep Learning Approach for Textile Fabric Material Identification from OCT and Macro Images'
    https://ieeexplore.ieee.org/abstract/document/10441457
"""
import torch
import torchvision


class TextileNetModel(torch.nn.Module):
    def __init__(self, model, pretrained=True):
        super(TextileNetModel, self).__init__()
        assert model is not None and model != '', 'Model must be provided.'
        if model == "vgg16":
            self.model = torchvision.models.vgg16(pretrained=pretrained)
            num_in_features = self.model.classifier[0].in_features
            self.model.classifier = self.get_custom_layers(num_in_features)
        elif model == "resnet101":
            self.model = torchvision.models.resnet101(pretrained=pretrained)
            num_in_features = self.model.fc.in_features
            self.model.fc = self.get_custom_layers(num_in_features)
        elif model == "inceptionv3":
            self.model = torchvision.models.inception_v3(pretrained=pretrained)
            num_in_features = self.model.fc.in_features
            self.model.fc = self.get_custom_layers(num_in_features)
        elif model == "mobilenetv2":
            self.model = torchvision.models.mobilenet_v2(pretrained=pretrained)
            num_in_features = self.model.classifier[1].in_features
            self.model.classifier = self.get_custom_layers(num_in_features)
        elif model == "convnextbase":
            self.model = torchvision.models.convnext_base(pretrained=pretrained)
            num_in_features = self.model.classifier[2].in_features
            self.model.classifier = self.get_custom_layers(num_in_features)
        else:
            assert False, "Model must be one of 'vgg16', 'resnet101', 'inceptionv3', 'mobilenetv2' or 'convnextbase'."

    def get_custom_layers(self, num_in_features):
        return torch.nn.Sequential(
            torch.nn.Linear(num_in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
            torch.nn.Softmax()
        )

    def forward(self, input):
        outputs = self.model(input)
        return outputs