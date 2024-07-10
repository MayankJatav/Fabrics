import torch
import torchvision.models as models

class MobileNetModel(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.feature_extractor = self.mobilenet.features
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(1280, 320),
            torch.nn.ReLU(),
            torch.nn.Linear(320, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 21),
            torch.nn.Softmax(),
        )

    def forward(self, input):
        x = self.feature_extractor(input)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classification(x)
        return x