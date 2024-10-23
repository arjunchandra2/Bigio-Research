import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class FilteringDataset(Dataset):
    def __init__(self, clean_dir, unclean_dir, transform=None):
        """
        Args:
            clean_dir (string): Directory with clean images.
            unclean_dir (string): Directory with unclean images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.clean_dir = clean_dir
        self.unclean_dir = unclean_dir
        self.clean_filenames = os.listdir(clean_dir)  # Clean images
        self.unclean_filenames = os.listdir(unclean_dir)  # Unclean images
        self.transform = transform

        # Create a combined list of all images, with corresponding labels
        self.image_filenames = [(os.path.join(clean_dir, f), 0) for f in self.clean_filenames] + \
                               [(os.path.join(unclean_dir, f), 1) for f in self.unclean_filenames]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path, label = self.image_filenames[idx]
        image = Image.open(img_path).convert("RGB")  # Open the image and convert to RGB
        
        if self.transform:
            image = self.transform(image)  # Apply any transformations

        return image, label  # Return the image and its label (0 for clean, 1 for unclean)
