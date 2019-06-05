import argparse
import copy
import cv2
import datetime as dt
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torchvision.models as m
import torchvision.transforms as transforms
from natsort import natsorted
import numpy as np
from PIL import ImageChops
from skimage import transform
from torch.utils.data import DataLoader, Dataset, random_split
from utils.pick_testset import pick_set


def main():
    # Architecture
    model = m.resnet50(pretrained=False)

    # Transforms
    transf = transforms.Compose([
        Rescale(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    dataset = FaceDataset(image_dir=data_path, transform=transf)
    dataset_size = len(dataset)

    if mode == 'train':
        print(model)
        model.load_state_dict(torch.load(weights_path))
        # Update requires_grad
        for param in model.parameters():
            param.requires_grad = False


        # Loss_fn, Optimizer, scheduler
        optimizer = optim.SGD(list(model.features[34].parameters()) + list(model.features[37].parameters()) +
                              list(model.features[40].parameters()) +
                              list(model.features[35].parameters()) + list(model.features[38].parameters()) +
                              list(model.features[41].parameters()) +
                              list(model.classifier[0].parameters()) + list(model.classifier[3].parameters()) +
                              list(model.classifier[6].parameters()), lr=lr)

        loss_fn = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Dataset
        train_length = int(dataset_size * 0.7)
        lengths = [train_length, dataset_size - train_length]
        train_set, val_set = random_split(dataset, lengths)
        dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
        dataloader = {
            'train': DataLoader(train_set, batch_size=batch_size, num_workers=workers),
            'val': DataLoader(val_set, batch_size=batch_size, num_workers=workers),
        }

        best_model, best_acc = train_model(model, dataloader, dataset_sizes, loss_fn, optimizer, scheduler, num_epochs)

        # Save best model
        now = dt.datetime.now()
        time_string = now.strftime('%m%d%H%M')
        fname = f'vgg-{time_string}-{str(int(best_acc * 10000))}.pth'
        torch.save(model.state_dict(), f'data/trained/{fname}')
        print(f'model saved as: {fname}')

    elif mode == 'test':
        if not weights_path:
            raise Exception('Weights are needed to test. Please provide weights file with --weights option')

        # TODO: model architecture

        print(f'loading model: {weights_path}')
        model.load_state_dict(torch.load(weights_path), strict=False)
        model.eval()

        # Dataset
        test_length = int(dataset_size * 0.1)
        lengths = [test_length, dataset_size - test_length]
        test_set, _ = random_split(dataset, lengths)
        dataset_sizes = {'test': len(test_set)}
        dataloader = {'test': DataLoader(test_set, batch_size=batch_size, num_workers=workers)}

        test_model(model, dataloader, dataset_sizes)


def train_model(model, dataloader, dataset_sizes, loss_fn, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.cpu().state_dict())
    best_acc = 0.0

    model = model.to(device)
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for data in dataloader[phase]:
                inputs, labels = data['image'], data['label']
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('[Epoch {}/{}] {} loss: {:.4f} Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.cpu().state_dict())
                model = model.to(device)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}\n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


def test_model(model, dataloader, dataset_sizes):
    model.eval()
    model = model.to(device)
    acc = 0.0
    with torch.set_grad_enabled(False):
        for data in dataloader['test']:
            inputs, labels, fnames = data['image'], data['label'], data['fname']
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)

            output = model(inputs)
            _, preds = torch.max(output, 1)
            acc += torch.sum(preds == labels.data)
            out = nn.functional.log_softmax(output, 1)
            res = [out[p] for p in preds]
            print(res)

            with open('output.txt', 'w') as f:
                for idx in range(len(fnames)):
                    fn = os.path.basename(fnames[idx])
                    f.write(f'{fn},{res[idx]:.7f}\n')
                    print('Outputs file saved as output.txt')

            with open('gt.txt', 'w') as f:
                for idx in range(len(fnames)):
                    fn = os.path.basename(fnames[idx])
                    f.write(f'{fn},{labels.data[idx]}\n')
                    print('Ground truth saved as gt.txt')

    acc = acc.double() / dataset_sizes['test']
    return acc


class Rescale(object):
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
    def __init__(self, image_dir, transform=None):
        self.images = []
        image_dirs = []
        label = []
        for c, l in dataset_classes:
            image_dirs.append(os.path.join(image_dir, c))
            label.append(l)

        for idx in range(len(image_dirs)):
            for img in natsorted(os.listdir(image_dirs[idx])):
                image = {}
                image['image'] = os.path.join(image_dirs[idx], img)
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
        return {'image': image, 'label': label, 'fname': sample['image']}


parser = argparse.ArgumentParser(description='Fake face classification')
parser.add_argument('-M', '--mode', help='running mode(test | train)', required=True)
parser.add_argument('--data', required=True,
                    help='path to dataset(ex. \'data/train\' where fake, gan, real are in data/train/')
parser.add_argument('-E', '--epoch', default=50, type=int, help='Num of epochs')
parser.add_argument('-B', '--batch', default=16, type=int, help='batch size')
parser.add_argument('--workers', type=int, required=True,
                    help='num of workers(check $ lscpu / take num core * num thread as input)')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate(decays over time)')
parser.add_argument('--schstep', default=8, type=int, help='Step size for SGD optimizer scheduler')
parser.add_argument('--gamma', default=0.9, type=float, help='Gamma (lr decay rate) for SGD optimizer scheduler')
parser.add_argument('--weights', nargs='?', help='weights to load(ex. data/model_params/vgg-0603-7890.pth')
args = parser.parse_args()

dataset_classes = [['fake', 0], ['real', 1], ['gan', 0]]
mode = args.mode
data_path = args.data  # default = data/
num_epochs = args.epoch
batch_size = args.batch  # default = 16
workers = args.workers
lr = args.lr  # default = 0.01
step_size = args.schstep
gamma = args.gamma
weights_path = args.weights  # optional
if mode not in ['train', 'test']:
    raise Exception('Running mode needed. Expected arguments are (train | test). See $ python main.py -h for help')
assert type(workers), int

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    main()
