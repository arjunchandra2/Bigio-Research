import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class DefectDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with defect images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)  
        self.transform = transform  

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")  
        
        if self.transform:
            image = self.transform(image) 

        return image  