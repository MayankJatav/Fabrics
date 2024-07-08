import glob
from torch.utils.data import Dataset
from torchvision.io import read_image

class FabricDataset(Dataset):
    def __init__(self, path, transform=None):
        # Initialization Code Here
        self.transform = transform
        self.files = glob.glob(path+"\\**\\**\\*.png", recursive=True)

    def __len__(self):
        # Reuturn  the length
        return len(self.files)

    def __getitem__(self, idx):
        # Return the data at index idx
        image = read_image(self.files[idx])
        if self.transform:
            image = self.transform(image)
        label = self.files[idx].split('\\')[-3]
        return image, label