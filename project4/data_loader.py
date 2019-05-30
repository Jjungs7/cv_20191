import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import transform
from natsort import natsorted


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img


class FaceDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        self.images = []
        for idx in range(len(image_dir)):
            for img in natsorted(os.listdir(image_dir[idx])):
                image = {}
                image['image'] = os.path.join(image_dir[idx], img)
                image['label'] = label[idx]
                self.images.append(image)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None

        sample = self.images[idx]
        image = cv2.imread(sample['image'], cv2.IMREAD_COLOR)
        label = torch.tensor(sample['label'], dtype=torch.int8)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label}