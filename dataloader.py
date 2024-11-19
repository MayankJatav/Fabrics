import glob
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from utils import Utilities as utils
from PIL import Image

class FabricDataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.files = glob.glob(path+"/**/**/*.png", recursive=True)
        self.files = [s.replace("\\", "/") for s in self.files]
        self.replace_words = {
                "Acrylic 75 Polyester 20Elastane 5": "Acrylic 75 Polyester 20 Elastane 5",
                "Cotton 100 Corduroy?": "Cotton 100",
                "Cotton 100?": "Cotton 100",
                "Cotton 100Elastane 1": "Cotton 100",
                "Cotton 50 Polyester 50?": "Cotton 50 Polyester 50",
                "Cotton 50Modal 50  0": "Cotton 50 Modal 50",
                "Cotton 50Polyester 50": "Cotton 50 Polyester 50",
                "Cotton 50Polyester 50?": "Cotton 50 Polyester 50",
                "Cotton 65Polyester 34": "Cotton 65 Polyester 34",
                "Cotton 80 Polyester 20?": "Cotton 80 Polyester 20",
                "Cotton 98Elastane 2": "Cotton 98 Elastane 2",
                "Cotton 99Elastane 1": "Cotton 99 Elastane 1",
                "Polyamide (Nylon) 95 Elastane 5 (Elastic net)": "Polyamide 95 Elastane 5",
                "Polyester 100 100  0": "Polyester 100",
                "Polyester 100T/R check 100": "Polyester 100",
                "Polyester 46Repreve 54": "Polyester 46 Repreve 54",
                "Polyester 65 Viscose 30 Elastane 5, Heavy and elastic crepe": "Polyester 65 Viscose 30 Elastane 5",
                "Viscose 100 100  0": "Viscose 100",
                "leather 100": "Leather 100"
            }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        image = Image.open(file)
        if self.transform:
            image = self.transform(image)
        image = image / torch.max(image)
        tag_path = file[:file.rindex('/')] + "/tag.txt"
        with open(tag_path, 'r') as f:
            composition = self.get_composition(f.readlines()[1].strip())
        label = [0] * len(utils.get_classes())
        for i in range(0, len(composition), 2):
            comp = composition[i]
            label[utils.class_to_index(comp)] = float(composition[i+1])
        label = torch.Tensor(label)
        return image, label

    def get_composition(self, str):
        if str in self.replace_words.keys():
            str = self.replace_words[str]
        l = str.split()
        composition = []
        word = ""
        for w in l:
            if w.isalpha():
                word += w + " "
            else:
                if utils.is_number(w):
                    composition.append(word.strip())
                    composition.append(w)
                    word = ""
        return composition

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