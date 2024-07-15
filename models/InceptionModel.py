import torch
import torchvision.models as models
import cv2
import numpy as np
from torch.nn import functional as F

class InceptionModel(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(InceptionModel, self).__init__()
        self.inception = Inception3()
        self.conv_vertical = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv_horizontal = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer= torch.nn.Sequential(
            torch.nn.Linear(352, 176),
            torch.nn.ReLU(),
            torch.nn.Linear(176, 88),
            torch.nn.ReLU(),
            torch.nn.Linear(88, 35),
            torch.nn.ReLU()
        )
        
    def forward(self, input):
        x = self.inception(input)
        vertical, horizontal = self.preprocess_input(input.detach().cpu().numpy())
        vertical = torch.unsqueeze(torch.from_numpy(vertical), 1)
        horizontal = torch.unsqueeze(torch.from_numpy(horizontal), 1)
        device = next(self.parameters()).device
        vertical = vertical.to(device)
        horizontal = horizontal.to(device)
        vertical = self.conv_vertical(vertical)
        vertical = self.avgpool(vertical)
        vertical = torch.flatten(vertical, 1)
        horizontal = self.conv_horizontal(horizontal)
        horizontal = self.avgpool(horizontal)
        horizontal = torch.flatten(horizontal, 1)
        cat = torch.concat([x, vertical, horizontal], dim=1)
        x = self.fc_layer(cat)
        return x

    def preprocess_input(self, image):
        x, y = [], []
        for i in image:
            i = np.transpose(i, axes=(1, 2, 0))
            i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            i = (i * 255).astype(np.uint8)
            i = cv2.GaussianBlur(i, (5, 5), cv2.BORDER_DEFAULT)
            sobel_x = cv2.Sobel(i, cv2.CV_8U, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(i, cv2.CV_8U, 0, 1, ksize=3)
            thresh, binary_sobel_x = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh, binary_sobel_y = cv2.threshold(sobel_y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            x.append(binary_sobel_x)
            y.append(binary_sobel_y)
        x = np.array(x).astype(np.float32)
        y = np.array(y).astype(np.float32)
        return x, y

class Inception3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        inception_blocks = [BasicConv2d, InceptionA]
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]

        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 288 x 1 x 1
        x = torch.flatten(x, 1)
        return x


class InceptionA(torch.nn.Module):
    def __init__(
        self, in_channels: int, pool_features: int, conv_block = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: torch.Tensor):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: torch.Tensor):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
    
class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)