import torch
import timm

class DeiTModel(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(DeiTModel, self).__init__()
        self.model = timm.create_model('deit_base_patch16_224', pretrained=pretrained)
        self.model.head = torch.nn.Sequential(
            torch.nn.Linear(768, 3),
            torch.nn.Softmax()
        )

    def forward(self, input):
        return self.model(input)