import torch
import torchvision.models as models

class InceptionModel(models.Inception3):
    def __init__(self, pretrained=False):
        super(InceptionModel, self).__init__()
        self.inception = models.inception_v3(pretrained=pretrained)
        self.inception.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1280),
            torch.nn.ReLU(),
            torch.nn.Linear(1280, 320),
            torch.nn.ReLU(),
            torch.nn.Linear(320, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 21),
            torch.nn.Softmax(),
        )

    def forward(self, input):
        x = self.inception(input)
        return x