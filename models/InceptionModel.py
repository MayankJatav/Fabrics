import torch
import torchvision.models as models

class InceptionModel(torch.nn.Module):
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
        if self.training:
            x = self.inception(input)[0]
        else:
            x = self.inception(input)
        print("Shape", input.shape, x.shape)
        return x