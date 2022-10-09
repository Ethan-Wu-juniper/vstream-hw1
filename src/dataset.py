import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import os
from config import Variables

class ImgDataset(Dataset):
    def __init__(self, set_type, images, label_df=None):
        self.images = images
        self.label_df = label_df
        self.set_type = set_type

    def __getitem__(self, idx):
        path = self.images[idx][0]
        img = self.images[idx][1]
        if self.label_df is None:
            return img, path
        else:
            label = self.label_df.loc[path].item()
            return img, label

    def __len__(self):
        return len(self.images)

def process_img(path):
    img = io.imread(path)
    img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
    img = img.to(dtype=torch.get_default_dtype()).div(255)
    return img

def load_dataset(set_type):
    label_df = None if set_type=="test" else pd.read_csv(f"{Variables.DATADIR}{set_type}.csv").set_index("names")
    img_path = os.listdir(f"{Variables.DATADIR}{set_type}")
    images = [(path, process_img(f"{Variables.DATADIR}{set_type}/{path}")) for path in img_path]
    return ImgDataset(set_type, images, label_df)

