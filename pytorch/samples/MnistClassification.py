import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
# from DynamicRELU import DYReLU2

tensorboard_on = False
if tensorboard_on:
    writer = SummaryWriter()


class MyConvNet(nn.Module):
    def __init__(self, relu, relustr, **kwargs):
        super(MyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        if relustr == 'dyrelu':
            self.relu1 = relu(20, 20)
            self.relu2 = relu(50, 50)
            self.relu3 = relu(500, 500)
        else:
            self.relu1 = relu()
            self.relu2 = relu()
            self.relu3 = relu()

    def forward(self, x):
        x = self.conv1(x)           # 28x28x1 -> 24x24x20
        x = self.relu1(x)           # 24x24x20
        x = F.max_pool2d(x, 2, 2)   # 24x24x20 -> 12x12x20
        x = self.conv2(x)           # 12x12x20 -> 8x8x50
        x = self.relu2(x)           # 8x8x50
        x = F.max_pool2d(x, 2, 2)   # 8x8x50 -> 4x4x50
        x = torch.flatten(x, 1)     # 4x4x50 -> 4*4*50
        x = self.fc1(x)             # 4*4*50 -> 500
        # x = self.relu3(x)           # 500 -> 500
        x = self.fc2(x)             # 500 -> 10
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = F.nll_loss(pred, target)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tensorboard_on:
            writer.add_scalar('Loss/train',
                              loss.item(),
                              epoch * len(train_loader) + batch_idx)

    # if batch_idx % 100 == 0:
    print("Epoch: {}, train loss: {}, ".format(epoch, loss.item()), end='')


def test(model, device, test_loader, epoch):
    model.eval()
    total_loss = 0
    correct = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset) * 100.
    print("test loss: {}, accuracy: {}".format(total_loss, acc))

    if tensorboard_on:
        writer.add_scalar('Loss/test', total_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)


def main():
    batch_size = 128
    lr = 0.01
    momentum = 0.9
    epochs = 15
    schd_step = 7
    relus = {'relu': nn.ReLU,
             'lrelu': nn.LeakyReLU,
             'rrelu': nn.RReLU,
             'prelu': nn.PReLU,
             'relu6': nn.ReLU6,
             'elu': nn.ELU,
             'selu': nn.SELU,
             # dyrelu': DYReLU2
            }
    relu_kwargs = [{}, {}, {}, {}, {}, {}, {}, {}]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    for i, (relustr, relu) in enumerate(relus.items()):
        print('--------------------- {} ---------------------'.format(relustr))
        model = MyConvNet(relu, relustr, **relu_kwargs[i]).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schd_step)

        for epoch in range(epochs):
            train(model, device, train_dataloader, optimizer, epoch)
            test(model, device, test_dataloader, epoch)
            scheduler.step()

        # torch.save(model.state_dict(), 'mnist_cnn.pt')


if __name__ == '__main__':
    main()

    if tensorboard_on:
        writer.close()
