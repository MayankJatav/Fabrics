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

    def __len__(self):
        # Reuturn  the length
        return len(self.files)

    def __getitem__(self, idx):
        # Return the data at index idx
        image = Image.open(self.files[idx])
        if self.transform:
            image = self.transform(image)
        image = image / torch.max(image)
        label = [0] * len(utils.get_classes())
        className = self.files[idx].split('/')[-3]
        label[utils.class_to_index(className)] = 1
        label = torch.Tensor(label)
        return image, label

class FabricOCTDataset(Dataset):
    def __init__(self, path, transform=None):
        # Initialization Code Here
        self.transform = transform
        self.files = glob.glob(path+"/**/**/*.png", recursive=True)
        self.files = [s.replace("\\", "/") for s in self.files]
    
    def __len__(self):
        # Reuturn  the length
        return len(self.files)

    def __getitem__(self, idx):
        # Return the data at index idx
        image = np.array(Image.open(self.files[idx]))
        image = self.preprocess_image(image)
        if self.transform:
            image = self.transform(image).float()
        label = [0, 0, 0]
        className = self.files[idx].split('/')[-3]
        label[utils.class_to_index(className)] = 1
        label = torch.Tensor(label)
        return image, label
    
    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) # Apply the sharpening kernel to the image
        image = cv2.filter2D(image, -1, sharpening_kernel)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image / 255
        return image