import copy
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torchvision.models as m
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from models.vgg_vd_16 import MyVgg
from data_loader import FaceDataset, Rescale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr, momentum, step_size, gamma = 0.1, 0.9, 10, 0.1
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def train_model(model, loss_fn, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.cpu().state_dict())
    best_acc = 0.0

    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        optimizer.step()
        model.train()

        running_loss = 0.0
        running_corrects = 0
        for data in dataloader['train']:
            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)
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
        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.cpu().state_dict())

            model = model.to(device)
        
        print()

        if epoch % 5 == 4:
            test_model(model)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model):
    model.eval()
    model = model.to(device)
    acc = 0.0
    for data in dataloader['test']:
        inputs, labels = data['image'], data['label']
        inputs = inputs.to(device)
        labels = labels.long().to(device)

        output = model(inputs)
        _, preds = torch.max(output, 1)
        acc += torch.sum(preds == labels.data)

    print('Accurracy of this model is : %.4f' % (acc.double() / dataset_sizes['test']))

# Model
model = m.vgg19_bn(pretrained=True)
#model = MyVgg(3, 2622)
#model = m.vgg16_bn(pretrained=False)
#model.load_state_dict(torch.load('data/model_params/vgg_face_dag_custom.pth'), strict=False)
#model.eval()
#model.classifier[6] = nn.Linear(model.classifier[6].in_features, 128, bias=True)
#model.classifier.add_module('classifier7', nn.Linear(128, 2, bias=True))
model = model.to(device)
#print(model)

# Transforms
scale = Rescale(224)
transf = transforms.Compose([
    scale,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
dataset = {t: FaceDataset(image_dir=[f'{DATA_DIR}/{t}/real', f'{DATA_DIR}/{t}/fake', f'{DATA_DIR}/{t}/gan'], label=[1, 0, 0], transform=transf) for t in ['train', 'test']}
dataloader = {t: DataLoader(dataset[t], batch_size=16, shuffle=True, num_workers=2) for t in ['train', 'test']}
dataset_sizes = {t: len(dataset[t]) for t in ['train', 'test']}

# Update requires_grad
#for param in model.parameters():
#    param.requires_grad = False
'''
model.features[40] = nn.Conv2d(512, 512, 3, 1, 1)
model.features[43] = nn.Conv2d(512, 512, 3, 1, 1)
model.features[46] = nn.Conv2d(512, 512, 3, 1, 1)
model.features[49] = nn.Conv2d(512, 512, 3, 1, 1)

model.classifier[0] = nn.Linear(25088, 4096, bias=True)
model.classifier[3] = nn.Linear(4096, 4096, bias=True)
model.classifier[6] = nn.Linear(4096, 1000, bias=True)
model.classifier.add_module('relu3', nn.ReLU(inplace=True))
model.classifier.add_module('dropout3', nn.Dropout())
model.classifier.add_module('fc4', nn.Linear(1000, 2, bias=True))
'''
# Loss_fn, Optimizer, scheduler
loss_fn = nn.CrossEntropyLoss()
'''
params = list(model.features[40].parameters()) + list(model.features[43].parameters()) + list(model.features[46].parameters()) + list(model.features[49].parameters())
params += list(model.classifier[0].parameters()) + list(model.classifier[3].parameters()) + list(model.classifier[6].parameters()) + list(model.classifier[9].parameters())

optimizer = optim.SGD(params, lr=lr)
'''
optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_model(model, loss_fn, optimizer, scheduler, 30)
