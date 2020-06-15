import os
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
from PIL import Image, ImageFilter


class PairedMNIST(MNIST):
    pairing_transformations = ["rotation", "blur"]
    pairing_criteria = ["same_number", "different_number"]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, pairing_criterion="random"):
        super(PairedMNIST, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        self.n_classes = len(self.classes)
        self.pairing_criterion = pairing_criterion
        self.class_indexes = {}
        for cl in range(10):
            self.class_indexes[cl] = np.where(self.targets == cl)[0]

    def __getitem__(self, index1):
        """
        Args:
            index (int): Index

        Returns:
            tuple: ((image1, image2), (target1, target2)) where target is index of the target class.
        """
        img1, target1 = self.data[index1], int(self.targets[index1])

        if self.pairing_criterion not in self.pairing_criteria:
            pairing_criterion = np.random.choice(self.pairing_criteria)
        else:
            pairing_criterion = self.pairing_criterion

        if pairing_criterion == "same_number":
            while True:
                index2 = np.random.choice(self.class_indexes[target1])
                if index2 != index1:
                    break
        elif pairing_criterion == "different_number":
            while True:
                target2 = np.random.randint(0, self.n_classes)
                if target2 != target1:
                    break
            index2 = np.random.choice(self.class_indexes[target2])
        else:
            while True:
                index2 = np.random.randint(0, len(self.data))
                if index2 != index1:
                    break

        img2, target2 = self.data[index2], int(self.targets[index2])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')

        if target2 != target1:
            transf = np.random.choice(self.pairing_transformations)
            if transf == "rotation":
                angle = np.random.choice([-15, 15])
                img1 = img1.rotate(angle)
                img2 = img2.rotate(angle)
            elif transf == "blur":
                img1 = img1.filter(ImageFilter.BLUR)
                img2 = img2.filter(ImageFilter.BLUR)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)

        return ((img1, img2), (target1, target2))

    def __len__(self):
        return len(self.data)


def test_paired_MNIST():
    import matplotlib.pyplot as plt
    dataset_root_folder = '/is/cluster/shared/ei/xray/trial_datasets'
    paired_dataset = PairedMNIST(dataset_root_folder, train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
    ]))
    paired_dataloader = torch.utils.data.DataLoader(
        paired_dataset, batch_size=4, shuffle=True)
    examples = enumerate(paired_dataloader)
    for idx, batch in examples:
        images, targets = batch
        images1_list, images2_list = images
        targets1_list, targets2_list = targets
        break
    
    for i in range(len(images1_list)):
        fig, ax = plt.subplots(1, 2, figsize=(10, 15))
        ax[0].imshow(images1_list[i][0, :, :])
        ax[0].set_title(targets1_list[i].item(), fontsize=20)
        ax[1].imshow(images2_list[i][0, :, :])
        ax[1].set_title(targets2_list[i].item(), fontsize=20)
        plt.show()


if __name__ == "__main__":
    test_paired_MNIST()
