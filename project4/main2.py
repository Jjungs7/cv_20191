import copy
import cv2
import datetime as dt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torchvision.models as m
import torchvision.transforms as transforms
from models.facenet import FaceNetModel
from natsort import natsorted
from skimage import transform
from torch.utils.data import DataLoader, Dataset
from utils.pick_testset import pick_set

def main():
    #mode = 'train'
    mode = 'test'
    global dataset_sizes

    # Transforms
    scale = Rescale(224)
    transf = transforms.Compose([
        scale,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if mode == 'train':
        # Model
        model = FaceNetModel(embedding_size=128, num_classes=2)

        # Update requires_grad
        for param in model.parameters():
            param.requires_grad = False

        model.model.layer4[2].conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        model.model.layer4[2].bn1 = nn.BatchNorm2d(512)
        model.model.layer4[2].conv2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        model.model.layer4[2].bn2 = nn.BatchNorm2d(512)
        model.model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        model.model.fc = nn.Linear(25088, batch_size, True)
        model.model.classifier = nn.Linear(batch_size, 2, True)

        # Loss_fn, Optimizer, scheduler
        loss_fn = nn.CrossEntropyLoss()

        lr = 0.05
        optimizer = optim.Adam(list(model.model.layer4[2].conv1.parameters()) + list(model.model.layer4[2].bn1.parameters()) +
                               list(model.model.layer4[2].conv2.parameters()) + list(model.model.layer4[2].bn2.parameters()) +
                               list(model.model.avgpool.parameters()) + list(model.model.fc.parameters()) +
                               list(model.model.classifier.parameters())
                               , lr=lr)

        # optimizer = optim.Adam(optimizer, lr=lr)
        model = model.to(device)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.15)

        # Dataset
        batch_size = 10
        dataset = {t: FaceDataset(image_dir=[f'{DATA_DIR}/{t}/real', f'{DATA_DIR}/{t}/fake', f'{DATA_DIR}/{t}/gan'], label=[1, 0, 0], transform=transf) for t in ['train', 'test', 'val']}
        dataloader = {t: DataLoader(dataset[t], batch_size=batch_size, shuffle=True, num_workers=6) for t in ['train', 'test', 'val']}
        dataset_sizes = {t: len(dataset[t]) for t in ['train', 'test', 'val']}

        train_model(model, dataloader, loss_fn, optimizer, scheduler, 50)
    elif mode == 'test':
        # Dataset
        best_model = 'vgg-123-0000.pth'
        for d in os.listdir('./data/model_params'):
            d_split = d.split('-')
            if len(d_split) == 3:
                best_model = d if d_split[2] > best_model.split('-')[2] else best_model
        model = FaceNetModel(embedding_size=128, num_classes=2)
        print(f'loading model: {best_model}')
        model.load_state_dict(torch.load(f'data/model_params/{best_model}'), strict=False)
        model.eval()
        batch_size = 32
        dataset = {t: FaceDataset(image_dir=[f'{DATA_DIR}/{t}/real', f'{DATA_DIR}/{t}/fake', f'{DATA_DIR}/{t}/gan'], label=[1, 0, 0], transform=transf) for t in ['test']}
        dataloader = {t: DataLoader(dataset[t], batch_size=batch_size, shuffle=True, num_workers=6) for t in ['test']}
        dataset_sizes = {t: len(dataset[t]) for t in ['test']}

        test_model(model, dataloader)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VAL_DIR = os.path.join(DATA_DIR, 'val')
classes = ['fake', 'real', 'gan']

train_dirs = [os.path.join(TRAIN_DIR, path) for path in classes]
test_dirs = [os.path.join(TEST_DIR, path) for path in classes]
val_dirs = [os.path.join(VAL_DIR, path) for path in classes]
for d in [DATA_DIR, TRAIN_DIR, TEST_DIR, VAL_DIR] + train_dirs + test_dirs + val_dirs:
    if not os.path.exists(d):
        print(f'Directory {d} created')
        os.mkdir(d)

for idx, direc in enumerate(test_dirs):
    for f in os.listdir(direc):
        os.rename(os.path.join(test_dirs[idx], f), os.path.join(os.path.join(train_dirs[idx], f)))

for idx, direc in enumerate(val_dirs):
    for f in os.listdir(direc):
        os.rename(os.path.join(val_dirs[idx], f), os.path.join(os.path.join(train_dirs[idx], f)))

pick_set(100, train_dirs, test_dirs)
pick_set(0, train_dirs, val_dirs)
dataset_sizes = {}


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
        return {'image': image, 'label': label, 'fname': sample['image']}


def train_model(model, dataloader, loss_fn, optimizer, scheduler, num_epochs=10):
    global dataset_sizes
    since = time.time()
    val_acc_history = []
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

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes['train']
            epoch_acc = running_corrects.double() / dataset_sizes['train']

            print('[Epoch {}/{}]Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            if epoch % 5 == 4 or epoch_acc > 0.9:
                best_model_wts, best_acc = validate_model(model, dataloader, best_model_wts, best_acc)
                model = model.to(device)

    best_model_wts, best_acc = test_model(model, best_model_wts, best_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}\n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloader):
    global dataset_sizes
    model.eval()
    model = model.to(device)
    acc = 0.0
    with torch.set_grad_enabled(False):
        for data in dataloader['test']:
            inputs, labels, fnames = data['image'], data['label'], data['fname']
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)

            output = model(inputs)
            print(output)
            _, preds = torch.max(output, 1)
            acc += torch.sum(preds == labels.data)
            res = [1 - max(output[i][0], output[i][1]) if labels.data[i] == 0 else max(output[i][0], output[i][1]) for i in range(len(fnames))]

            with open('output.txt', 'w') as f:
                for idx in range(len(fnames)):
                    fn = os.path.basename(fnames[idx])
                    f.write(f'{fn},{res[idx]:.7f}\n')

            with open('ground_truth.txt', 'w') as f:
                for idx in range(len(fnames)):
                    fn = os.path.basename(fnames[idx])
                    f.write(f'{fn},{labels.data[idx]}\n')

    acc = acc.double() / dataset_sizes['test']
    return acc

def validate_model(model, dataloader, best_model_wts, best_acc):
    global dataset_sizes
    model.eval()
    model = model.to(device)
    acc = 0.0
    with torch.set_grad_enabled(False):
        for data in dataloader['test']:
            inputs, labels = data['image'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)

            output = model(inputs)
            _, preds = torch.max(output, 1)
            acc += torch.sum(preds == labels.data)

    cur_acc = acc.double() / dataset_sizes['val']

    if cur_acc > best_acc:
        now = dt.datetime.now()
        time_string = now.strftime('%m_%d %H_%M')

        best_acc = cur_acc
        best_model_wts = copy.deepcopy(model.cpu().state_dict())
        fname = f'vgg-{time_string}-{str(int(best_acc * 10000))}.pth'
        torch.save(model.state_dict(), f'data/model_params/{fname}')
        print(f'model saved as: {fname}')

    print('Accuracy of this model is : %.4f' % (acc.double() / dataset_sizes['test']))
    return best_model_wts, best_acc


if __name__=="__main__":
    main()

