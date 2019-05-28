import numpy as np
import os
from PIL import Image
import torch
import torch.cuda
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

print(torch.cuda.is_available())


class FaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.images = natsorted(os.listdir(image_dir))
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None

        print(self.images[idx])
        return Image.open(os.path.join(self.image_dir, self.images[idx]))


# Model
model = models.vgg19_bn(pretrained=True)

# Transforms
transf = transforms.Compose([transforms.Resize((224, 224))])

# Dataset
face_fake_dataset = FaceDataset(image_dir=f'{DATA_DIR}/train/fake', transform=transf)
face_gan_dataset = FaceDataset(image_dir=f'{DATA_DIR}/train/gan', transform=transf)
face_real_dataset = FaceDataset(image_dir=f'{DATA_DIR}/train/real', transform=transf)

face_fake_dataset[0].save('test1.png')
face_gan_dataset[0].save('test2.png')
face_real_dataset[0].save('test3.png')

print(model)