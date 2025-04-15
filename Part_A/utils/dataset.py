import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class iNaturalist(Dataset):
    """
        Class representing the iNaturalist dataset.

        Args : 
            img_path : path to the csv file
            transforms : transformation that will be applied on the images

        Returns:
            img : image 
            label : corresponding label of the image    
    
    """
    def __init__(self, img_path, transforms=None):
        self.img_path_csv = pd.read_csv(img_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.img_path_csv)    
    
    def __getitem__(self, idx):
        img_path = self.img_path_csv.iloc[idx, 0]
        # Reading image
        img = Image.open(img_path).convert('RGB')
        label = self.img_path_csv.iloc[idx, 1]
        if self.transforms:
            img = self.transforms(img)

        return img, label    
       


