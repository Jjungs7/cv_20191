import argparse
import copy
import cv2
import datetime as dt
import os
import shutil
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
from PIL import Image, ImageChops
import signal
from skimage import transform
from torch.utils.data import DataLoader, Dataset, random_split


def main():
    # Architecture
    model = m.resnet50(pretrained=False)

    # Transforms
    transf = transforms.Compose([
        Rescale(224),
        RandomFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    if not refresh_ela:
        with open('data/last_ela.txt', 'r') as f:
            ela_dict = json.loads(f)
        dataset = FaceDataset(ela_dir=ela_dict)
    else:
        dataset = FaceDataset(image_dir=data_path, transform=transf)
    dataset_size = len(dataset)

    if mode == 'train':
        if weights_path.find('resnet50_ft_dag.pth') >= 0:
            model.fc = nn.Linear(2048, 8631, bias=True)
        else:
            model.fc = nn.Linear(2048, 3, bias=True)

        model.load_state_dict(torch.load(weights_path))
        # Update requires_grad
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

        if weights_path.find('resnet50_ft_dag.pth') >= 0:
            model.fc = nn.Linear(2048, 3, bias=True)

        # Loss_fn, Optimizer, scheduler
        params = list(model.layer4.parameters()) + list(model.fc.parameters())

        optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=decay)
        loss_fn = nn.CrossEntropyLoss()

        # Dataset
        train_length = int(dataset_size * 0.9)
        lengths = [train_length, dataset_size - train_length]
        train_set, val_set = random_split(dataset, lengths)
        dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
        dataloader = {
            'train': DataLoader(train_set, batch_size=batch_size, num_workers=workers),
            'val': DataLoader(val_set, batch_size=batch_size, num_workers=workers),
        }

        best_model, best_acc = train_model(model, dataloader, dataset_sizes, loss_fn, optimizer, num_epochs)

        # Save best model
        now = dt.datetime.now()
        time_string = now.strftime('%m%d%H%M')
        fname = f'resnet-{time_string}-{str(int(best_acc * 10000))}.pth'
        torch.save(best_model.state_dict(), f'data/trained/{fname}')
        print(f'model saved as: {fname}')

    elif mode == 'test':
        if not weights_path:
            raise Exception('Weights are needed to test. Please provide weights file with --weights option')

        # TODO: model architecture
        model.fc = nn.Linear(2048, 3, bias=True)

        print(f'loading model: {weights_path}')
        model.load_state_dict(torch.load(weights_path), strict=False)
        model.eval()

        # Dataset
        test_length = int(dataset_size)
        lengths = [test_length, dataset_size - test_length]
        test_set, _ = random_split(dataset, lengths)
        dataset_sizes = {'test': len(test_set)}
        dataloader = {'test': DataLoader(test_set, batch_size=batch_size, num_workers=workers)}

        test_model(model, dataloader, dataset_sizes)


def train_model(model, dataloader, dataset_sizes, loss_fn, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.cpu().state_dict())
    best_acc = 0.0
    last_val_loss = 10.0

    model = model.to(device)
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'val' and dataset_sizes['val'] <= 0:
                continue

            if phase == 'train':
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
                    # l1_crit = nn.L1Loss(False)
                    # reg_loss = 0
                    # for param in model.parameters():
                    #     reg_loss += l1_crit(param)

                    if phase == 'train':
                        # if abs(last_val_loss - running_loss) > 0.1:
                        #     loss += 0.0005 * reg_loss
                        loss.backward()
                        optimizer.step()
                    if phase == 'val':
                        last_val_loss = loss


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('[Epoch {}/{}] {} loss: {:.4f} Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.cpu().state_dict())

                model = model.to(device)

            if phase == 'val' and epoch % 25 == 24:
                # Save best model
                now = dt.datetime.now()
                time_string = now.strftime('%m%d%H%M')
                fname = f'resnet-{time_string}-{str(int(best_acc * 10000))}.pth'
                model.load_state_dict(best_model_wts)
                torch.save(model.state_dict(), f'data/trained/{fname}')
                print(f'model saved as: {fname}')

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
            out = nn.functional.sigmoid(output)
            _, preds = torch.max(out, 1)
            acc += torch.sum(preds == labels.data)
            vals = []
            for i in range(len(preds)):
                res = out[i, preds[i]]
                if preds[i] == 0:
                    greater = max(out[i, 1], out[i, 2])
                    v = res + greater
                    vals.append(res / v)
                else:
                    v = res + out[i, 0]
                    vals.append(1 - (res / v))

            with open('prob.txt', 'a') as f:
                for idx in range(len(fnames)):
                    fn = os.path.basename(fnames[idx])
                    f.write(f'{fn},{vals[idx]:.7f}\n')
                print('Probabilities file saved as prob.txt')

            with open('output.txt', 'a') as f:
                for idx in range(len(fnames)):
                    fn = os.path.basename(fnames[idx])
                    f.write(f'{fn},{preds[idx]:.7f}\n')
                print('Outputs file saved as output.txt')

            with open('gt.txt', 'a') as f:
                for idx in range(len(fnames)):
                    fn = os.path.basename(fnames[idx])
                    f.write(f'{fn},{labels.data[idx]}\n')
                print('Ground truth saved as gt.txt')

    acc = acc / dataset_sizes['test']
    return acc


class RandomFlip(object):
    def __init__(self, prob):
        assert isinstance(prob, float)
        self.prob = prob

    def __call__(self, sample):
        image = sample
        if np.random.rand() <= self.prob:
            np.flipud(image)

        return image


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
    def __init__(self, image_dir='', transform=None, ela_dict=None):
        if image_dir and not ela_dict:
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

            if mode == 'train' and refresh_ela == True:
                import json
                with open('data/last_ela.txt', 'w') as f:
                    f.write(json.dumps(self.images))

            self.transform = transform
        elif not image_dir and ela_dict:
            self.images = ela_dict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None

        sample = self.images[idx]
        if mode == 'train':
            false_name = f'data/ela/temp{idx}.jpg'
        else:
            false_name = f'data/test_ela/temp{idx}.jpg'
        if not os.path.exists(false_name):
            im_original = Image.open(sample['image'])
            im_original.save(false_name, quality=90)
            im_false = Image.open(false_name)
            diff = ImageChops.difference(im_original, im_false)
            d = diff.load()
            w, h = diff.size
            for x in range(w):
                for y in range(h):
                    d[x, y] = tuple(k * 20 for k in d[x, y])
            diff.save(false_name)

        image = cv2.imread(false_name, cv2.IMREAD_COLOR)
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
parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate(decays over time)')
parser.add_argument('--decay', default=0.01, type=float, help='Weight decay for the optimizer')
parser.add_argument('--momentum', default=0.9, type=float, help='Weight decay for the optimizer')
parser.add_argument('--weights', nargs='?', help='weights to load(ex. data/model_params/vgg-0603-7890.pth')
parser.add_argument('--refela', default=True, type=bool, help='False if you want to use ela of previous run')
args = parser.parse_args()

dataset_classes = [['real', 0], ['fake', 1], ['gan', 2]]
mode = args.mode
data_path = args.data  # default = data/
num_epochs = args.epoch
batch_size = args.batch  # default = 16
workers = args.workers
lr = args.lr  # default = 0.01
decay = args.decay
momentum = args.momentum
weights_path = args.weights  # optional
refresh_ela = args.refela
if mode not in ['train', 'test']:
    raise Exception('Running mode needed. Expected arguments are (train | test). See $ python main.py -h for help')
assert type(workers), int

if mode == 'test':
    if os.path.exists('prob.txt'):
        os.remove('prob.txt')

    if os.path.exists('output.txt'):
        os.remove('output.txt')

    if os.path.exists('gt.txt'):
        os.remove('gt.txt')

    if os.path.exists('data/test_ela'):
        shutil.rmtree('data/test_ela')
        os.mkdir('data/test_ela')

if mode == 'train' and refresh_ela and os.path.exists('data/ela'):
    shutil.rmtree('data/ela')
    os.mkdir('data/ela')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    main()
