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
lr, momentum, step_size, gamma = 0.01, 0.9, 10, 0.1
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

#def train_model(model, loss_fn, optimizer, scheduler, num_epochs=10):
def train_model(model, loss_fn, optimizer, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.cpu().state_dict())
    best_acc = 0.0

    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for data in dataloader[phase]:
                inputs, labels = data['image'], data['label']
                inputs = inputs.to(device)
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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.cpu().state_dict())

                model = model.to(device)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Model
#model = m.vgg19_bn(pretrained=True)
#model = MyVgg(3, 2622)
model = m.vgg16_bn(pretrained=False)
model.classifier[6] = nn.Linear(in_features=4096, out_features=2622, bias=True)
model.load_state_dict(torch.load('data/model_params/vgg_face_dag_custom.pth'), strict=False)
model.eval()

# Transforms
scale = Rescale(224)
transf = transforms.Compose([
    scale,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
dataset = {t: FaceDataset(image_dir=[f'{DATA_DIR}/{t}/real', f'{DATA_DIR}/{t}/fake', f'{DATA_DIR}/{t}/gan'], label=[1, 0, 0], transform=transf) for t in ['train', 'test']}

#dataset = {t: {x: FaceDataset(image_dir=f'{DATA_DIR}/{t}/{x}', label=1 if x == 'real' else 0, transform=transf) for x in ['real', 'fake', 'gan']} for t in ['train', 'test']}

#dataloader = {t: {x: DataLoader(dataset[t][x], batch_size=16, shuffle=True, num_workers=2) for x in ['real', 'fake', 'gan']} for t in ['train', 'test']}
dataloader = {t: DataLoader(dataset[t], batch_size=32, shuffle=True, num_workers=4) for t in ['train', 'test']}

dataset_sizes = {t: len(dataset[t]) for t in ['train', 'test']}

# Update requires_grad
for param in model.parameters():
    param.requires_grad = False

model.classifier[0] = nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features, bias=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, model.classifier[3].out_features, bias=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2, bias=True)
model = model.to(device)

# Loss_fn, Optimizer, scheduler
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(list(model.classifier[0].parameters()) + list(model.classifier[3].parameters()) +
                      list(model.classifier[6].parameters()), lr=lr, momentum=momentum)

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

#train_model(model, loss_fn, optimizer, scheduler, 20)
train_model(model, loss_fn, optimizer, 20)
