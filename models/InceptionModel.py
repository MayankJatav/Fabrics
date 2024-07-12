import torch
import torchvision.models as models
import cv2
import numpy as np

class InceptionModel(models.Inception3):
    def __init__(self, pretrained=False):
        super(InceptionModel, self).__init__()
        self.fc_layer= torch.nn.Sequential(
            torch.nn.Linear(6144, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1280),
            torch.nn.ReLU(),
            torch.nn.Linear(1280, 320),
            torch.nn.ReLU(),
            torch.nn.Linear(320, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 35),
            torch.nn.ReLU(),
        )
        self.conv_vertical = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU()
        )
        self.conv_horizontal = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU()
        )
        self.vertical_fc = torch.nn.Sequential(
            torch.nn.Linear(89401, 2048),
            torch.nn.ReLU()
        )
        self.horizontal_fc = torch.nn.Sequential(
            torch.nn.Linear(89401, 2048),
            torch.nn.ReLU()
        )


    def forward_inception(self, x):
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
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x, aux
        
    def forward(self, input):
        x, aux = self.forward_inception(input)
        vertical, horizontal = self.preprocess_input(input.detach().cpu().numpy())
        vertical = torch.unsqueeze(torch.from_numpy(vertical), 1)
        horizontal = torch.unsqueeze(torch.from_numpy(horizontal), 1)
        device = next(self.parameters()).device
        vertical = vertical.to(device)
        horizontal = horizontal.to(device)
        vertical = self.conv_vertical(vertical)
        vertical = torch.flatten(vertical, 1)
        vertical = self.vertical_fc(vertical)
        horizontal = self.conv_horizontal(horizontal)
        horizontal = torch.flatten(horizontal, 1)
        horizontal = self.horizontal_fc(horizontal)
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