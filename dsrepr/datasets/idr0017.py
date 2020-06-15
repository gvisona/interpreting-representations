import numpy as np
import zarr
import torch
import torchvision
import os
from skimage import io
from sklearn.model_selection import train_test_split
from PIL import Image

class IDR0017_FullImgs_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_directory, training=True, transform=None):
        super(IDR0017_FullImgs_Dataset, self).__init__()
        self.img_directory = img_directory
        self.training = training
        self.transform = transform
        img_paths = []
        for root, dirs, files in os.walk(img_directory):
            for fname in files:
                if fname.endswith('.jpg'):
                    dcr = fname.split(".")[0].split("-")
                    img_paths.append((os.path.join(root, fname), dcr))
        assert len(img_paths) > 0, "No images found in path"

        train_data, test_data = train_test_split(
            img_paths, test_size=0.1, random_state=42, shuffle=True)
        self.data = train_data if training else test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        impath, dcr = self.data[index]
        drug, cell_line, replicate = dcr
        image = io.imread(impath, plugin="pil")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "drug": drug, "cell_line": cell_line, "replicate": replicate}


class IDR0017_FullImgs_ZarrDataset(torch.utils.data.Dataset):
    def __init__(self, img_directory, whole_dataset=False, training=True, transform=None):
        super(IDR0017_FullImgs_ZarrDataset, self).__init__()
        self.img_directory = img_directory
        self.training = training
        self.transform = transform
        img_dataset = zarr.open(img_directory, mode="r")
        img_array = img_dataset["images"]
        assert len(img_array) > 0, "No images found in path"

        self.data = img_array
        if whole_dataset:
            self.idxs = list(range(len(img_array)))
        else:
            train_idxs, test_idxs = train_test_split(
                list(range(len(img_array))), test_size=0.1, random_state=42, shuffle=True)
            self.idxs = train_idxs if training else test_idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        lookup_idx = self.idxs[index]
        image = self.data[lookup_idx]
        if self.transform is not None:
            image = self.transform(image)
        return torch.tensor(image)

class IDR0017_SegmentedImgs_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_directory, training=True, transform=None):
        super(IDR0017_SegmentedImgs_Dataset, self).__init__()
        self.img_directory = img_directory
        self.training = training
        self.transform = transform
        img_paths = []
        for root, dirs, files in os.walk(img_directory):
            for fname in files:
                if fname.endswith('.jpg'):
                    dcr = fname.split(".")[0].split("-")
                    img_paths.append((os.path.join(root, fname), dcr))
        assert len(img_paths) > 0, "No images found in path"

        train_data, test_data = train_test_split(
            img_paths, test_size=0.1, random_state=42, shuffle=True)
        self.data = train_data if training else test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        impath, dcr = self.data[index]
        drug, cell_line, replicate = dcr
        image = io.imread(impath, plugin="pil")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "drug": drug, "cell_line": cell_line, "replicate": replicate}

if __name__ == "__main__":
    idr = IDR0017_FullImgs_Dataset(
        "/media/gvisona/GioData/idr0017/data/images")
