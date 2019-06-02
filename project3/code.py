import copy
import datetime as dt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

batch_size, num_workers, lr, momentum, num_epochs = 32, 4, 0.01, 0.9, 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = x.view(x.size(0), 16 * 12 * 12)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class FCN(nn.Module):
    def __init__(self, in_f, height, width, out_f):
        super(FCN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(784, 112),
            nn.LeakyReLU(),
            nn.Linear(112, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.sequential(x)


def train_model(model_name, model, dataloader, loss_fn, optimizer, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.cpu().state_dict())
    best_acc = 0.0

    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()

        running_loss = 0.0
        running_corrects = 0
        for data in dataloader['train']:
            inputs, labels = data[0], data[1]
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

        print('Loss({}): {:.4f} Acc: {:.4f}'.format(model_name, epoch_loss, epoch_acc))
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
        inputs, labels = data[0], data[1]
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
        fname = f'vgg_{time_string}_{str(int(best_acc * 10000))}.pth'
        torch.save(model.state_dict(), f'data/model_params/{fname}')
        print(f'model saved as: {fname}')

    print('Accurracy of this model is : %.4f' % (acc.double() / dataset_sizes['test']))
    print()

    return best_model_wts, best_acc


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
    'test': torch.utils.data.DataLoader(testset, batch_size=32, num_workers=num_workers)
}
dataset_sizes = {
    'train': len(trainset),
    'test': len(testset)
}

model_fcn = FCN(1, 28, 28, 10)
model_fcn = model_fcn.to(device)
model_lenet = LeNet()
model_lenet = model_lenet.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer_fcn = optim.Adam(model_fcn.parameters(), lr=lr)
optimizer_lenet = optim.Adam(model_lenet.parameters(), lr=lr)

train_model('fcn', model_fcn, dataloader, loss_fn, optimizer_fcn, num_epochs=num_epochs)
train_model('lenet', model_lenet, dataloader, loss_fn, optimizer_lenet, num_epochs=num_epochs)
