import glob
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from utils import Utilities as utils
from PIL import Image
import numpy as np
import cv2

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

class TextileNetDataset(Dataset):
    def __init__(self, path, transform=None):
        # Initialization Code Here
        self.transform = transform
        self.files = glob.glob(path+"/denim/*.jpg", recursive=True)
        self.files.extend(glob.glob(path+"/denim/*.jpeg", recursive=True))
        self.files.extend(glob.glob(path+"/satin/*.jpg", recursive=True))
        self.files.extend(glob.glob(path+"/satin/*.jpeg", recursive=True))
        self.files.extend(glob.glob(path+"/knit/*.jpg", recursive=True))
        self.files.extend(glob.glob(path+"/knit/*.jpeg", recursive=True))
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
        label = [0, 0, 0]
        className = self.files[idx].split('/')[-2]
        # print(className)
        label[class_to_index(className)] = 1
        label = torch.Tensor(label)
        return image, label

def get_fabrics_dataset_dataloder(dataset_path, transform, train_size, val_size, batch_size):
    dataset = FabricDataset(dataset_path, transform)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
    
def get_fabrics_oct_dataset_dataloder(dataset_path, transform, train_size, val_size, batch_size):
    dataset = FabricDataset(dataset_path, transform)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
    
def get_textilenet_dataset_dataloder(dataset_path, transform, train_size, val_size, batch_size):
    train_dataset = FabricDataset(dataset_path + "/train", transform)
    val_dataset = FabricDataset(dataset_path + "/test", transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader