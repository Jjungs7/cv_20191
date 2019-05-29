import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

batch_size, num_workers, lr, momentum, num_epochs = 32, 4, 0.01, 0.9, 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FCN(nn.Module):
    def __init__(self, in_f, height, width, out_f):
        super(FCN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_f, 28, 3, 1, 1),
            nn.Conv2d(28, 56, 3, 1, 1),
            nn.Conv2d(56, 56, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(56)
        )
        self.max_pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(25088, 14)
        self.fc2 = nn.Linear(14, out_f)

    def forward(self, x):
        x = self.sequential(x)
        x = self.max_pool1(x)
        x = x.view(-1, 25088)
        print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=20):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



transf = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transf)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transf)
classes = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

dataloader = {
    'train': torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=num_workers),
    'test': torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=num_workers)
}

model_fcn = FCN(1, 28, 28, 10)
loss_fn = nn.L1Loss(reduction='mean')
optimizer = optim.Adam(model_fcn.parameters(), lr=lr)

train_model(model_fcn, dataloader, loss_fn, optimizer, num_epochs=num_epochs)