from torchvision.models import vit_b_16
import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(3, 8, 3, stride=3)
        self.cnn2 = nn.Conv2d(8, 16, 3, stride=3)
        self.cnn3 = nn.Conv2d(16, 32, 3, stride=3)
        self.cnn4 = nn.Conv2d(32, 64, 3, stride=3)
        self.cnn5 = nn.Conv2d(64, 128, 2, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.cnn4(x)
        x = self.relu(x)
        x = self.cnn5(x)
        x = self.relu(x)
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
        x2 = x2.flatten(1)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x)
        return x