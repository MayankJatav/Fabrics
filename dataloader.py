import glob
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from utils import Utilities as utils
from PIL import Image

class FabricDataset(Dataset):
    def __init__(self, path, transform=None):
        # Initialization Code Here
        self.transform = transform
        self.files = glob.glob(path+"/**/**/*.png", recursive=True)
        self.files = [s.replace("\\", "/") for s in self.files]
        self.utils = utils()

    def __len__(self):
        # Reuturn  the length
        return len(self.files)

    def __getitem__(self, idx):
        # Return the data at index idx
        image = Image.open(self.files[idx])
        if self.transform:
            image = self.transform(image)
        image = image / torch.max(image)
        label = [0] * len(self.utils.classes)
        className = self.files[idx].split('/')[-3]
        label[self.utils.class_to_index(className)] = 1
        label = torch.Tensor(label)
        return image, label