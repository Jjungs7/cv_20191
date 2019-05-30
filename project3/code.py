import copy
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


def train_model(model, data, loss_fn, optimizer, num_epochs=20):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(data.dataset)
        epoch_acc = running_corrects.double() / len(data.dataset)
        best_acc = max(epoch_acc, best_acc)

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, data):
    acc = 0.0
    for inputs, labels in data:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        acc += torch.sum(preds == labels.data)

    print('Accuracy on Test set: {}'.format(acc.item() / len(data.dataset)))

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

model_fcn = FCN(1, 28, 28, 10)
model_fcn = model_fcn.to(device)
model_lenet = LeNet()
model_lenet = model_lenet.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer_fcn = optim.Adam(model_fcn.parameters(), lr=lr)
optimizer_lenet = optim.Adam(model_lenet.parameters(), lr=lr)

#train_model(model_fcn, dataloader['train'], loss_fn, optimizer_fcn, num_epochs=num_epochs)
train_model(model_lenet, dataloader['train'], loss_fn, optimizer_lenet, num_epochs=num_epochs)

#test_model(model_fcn, dataloader['test'])
test_model(model_lenet, dataloader['test'])