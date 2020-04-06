import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
    
    def forward(self, x):
        x = self.conv1(x)           # 28x28 -> 24x24
        x = F.relu(x)               # 24x24
        x = F.max_pool2d(x, 2, 2)   # 24x24 -> 12x12
        x = self.conv2(x)           # 12x12 -> 8x8
        x = F.relu(x)               # 8x8
        x = F.max_pool2d(x, 2, 2)   # 8x8 -> 4x4
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
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

        if batch_idx % 100 == 0:
            print("Train epoch: {}, iteration: {}, Loss: {}".format(
                epoch, batch_idx, loss.item()
            ))

def test(model, device, test_loader):
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
    print("Test loss: {}, accuracy: {}".format(total_loss, acc))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    lr = 1e-2
    momentum = 0.5
    epochs = 10

    model = MyConvNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        train(model, device, train_dataloader, optimizer, epoch)
        test(model, device, test_dataloader)

    torch.save(model.state_dict(), 'mnist_cnn.pt')

if __name__ == '__main__':
    main()
