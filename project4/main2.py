import copy
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
from torch.utils.data import DataLoader
from data_loader import FaceDataset, Rescale
from models.facenet import FaceNetModel
from utils.pick_testset import pick_set

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size, lr, momentum, step_size, gamma = 32, 0.05, 0.9, 10, 0.1
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALIDATION_PERCENTAGE = 30
classes = ['fake', 'real', 'gan']

train_dirs = [os.path.join(TRAIN_DIR, path) for path in classes]
test_dirs = [os.path.join(TEST_DIR, path) for path in classes]
for idx, direc in enumerate(test_dirs):
    for f in os.listdir(direc):
        os.rename(os.path.join(test_dirs[idx], f), os.path.join(os.path.join(train_dirs[idx], f)))

pick_set(VALIDATION_PERCENTAGE, train_dirs, test_dirs)


def train_model(model, loss_fn, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.cpu().state_dict())
    best_acc = 0.0

    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        scheduler.step()
        model.train()

        running_loss = 0.0
        running_corrects = 0
        for data in dataloader['train']:
            inputs, labels = data['image'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        print()

        if epoch % 5 == 4 or epoch_acc > 0.9:
            best_model_wts, best_acc = test_model(model, best_model_wts, best_acc)

    best_model_wts, best_acc = test_model(model, best_model_wts, best_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, best_model_wts, best_acc):
    model.eval()
    model = model.to(device)
    acc = 0.0
    for data in dataloader['test']:
        inputs, labels = data['image'], data['label']
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)

        output = model(inputs)
        _, preds = torch.max(output, 1)
        acc += torch.sum(preds == labels.data)

    cur_acc = acc.double() / dataset_sizes['test']

    # deep copy the model
    if cur_acc > best_acc:
        now = dt.datetime.now()
        time_string = now.strftime('%m-%d_%H-%M')

        best_acc = cur_acc
        best_model_wts = copy.deepcopy(model.cpu().state_dict())
        fname = f'facenet_{time_string}_{str(int(best_acc * 10000))}.pth'
        torch.save(model.state_dict(), f'data/model_params/{fname}')
        print(f'model saved as: {fname}')

        model = model.to(device)

    print('Accurracy of this model is : %.4f' % (acc.double() / dataset_sizes['test']))
    print()

    return best_model_wts, best_acc


# Model
model = FaceNetModel(embedding_size=128, num_classes=2)

# Transforms
scale = Rescale(224)
transf = transforms.Compose([
    scale,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
dataset = {t: FaceDataset(image_dir=[f'{DATA_DIR}/{t}/real', f'{DATA_DIR}/{t}/fake', f'{DATA_DIR}/{t}/gan'], label=[1, 0, 0], transform=transf) for t in ['train', 'test']}
dataloader = {t: DataLoader(dataset[t], batch_size=batch_size, shuffle=True, num_workers=12) for t in ['train', 'test']}
dataset_sizes = {t: len(dataset[t]) for t in ['train', 'test']}

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

# optimizer = optim.Adam(list(model.model.layer4[2].conv1.parameters()) + list(model.model.layer4[2].bn1.parameters()) +
#                        list(model.model.layer4[2].conv2.parameters()) + list(model.model.layer4[2].bn2.parameters()) +
#                        list(model.model.avgpool.parameters()) + list(model.model.fc.parameters()) +
#                        list(model.model.classifier.parameters())
#                        , lr=0.001)

optimizer = optim.Adam(model.parameters(), lr=0.001)
model = model.to(device)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_model(model, loss_fn, optimizer, scheduler, 50)
