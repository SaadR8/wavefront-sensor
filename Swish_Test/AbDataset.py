from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


class AbDataset(Dataset):
    def __init__(self, transform=None):
        self.zernikes = np.load("PATH/NetInput.npy")
        self.ID = range(0, 1000)
        self.transform = transform

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, idx):

        index = self.ID[idx]
        pattern = Image.open(
            "PATH/Image{0}.tif".format(index))

        pattern = TF.affine(pattern, -6, (-50, 0), 1, 0)

        zernike = self.zernikes[idx]

        if self.transform:
            pattern = self.transform(pattern)

        return pattern, zernike
