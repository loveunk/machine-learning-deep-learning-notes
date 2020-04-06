import torch
from torchvision import models, datasets, transforms
import copy

def set_parameter_requires_grad(model, feature_extract):
    for param in model.parameters():
        param.requires_grad = feature_extract

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == 'resnet':
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224
        return model_ft, input_size
    else:
        raise NotImplementedError

def dataloader(batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    return {'train':train_loader, 'test':test_loader}

def train_model(model, epochs, batch_size, loss_fn, optimizer, device, dataloaders):
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        for phase in ['train', 'test']:
            running_loss, running_corrects = 0., 0.
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for data, target in dataloaders[phase]:
                data, targets = data.to(device), target.to(device)
                # resnet accepts images with 3 channels
                data = data.repeat(1,3,1,1)

                with torch.autograd.set_grad_enabled(phase == 'train'):
                    outputs = model(data)
                    loss = loss_fn(outputs, targets)

                preds = outputs.argmax(dim=1)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(preds.cpu().view(-1) == targets.cpu().view(-1))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc  = running_corrects / len(dataloaders[phase].dataset)

            print('Epoch: {}, phase: {}, loss: {}, acc: {}'.format(epoch, phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def main():
    classes = 10
    epochs = 10
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_ft, input_size = initialize_model('resnet', classes,
                                            feature_extract=True,
                                            use_pretrained=True)
    # print(model_ft.fc.weight.requires_grad)

    dataloaders = dataloader(batch_size)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, 
                                       model_ft.parameters()), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_model(model_ft.to(device), epochs, batch_size, 
                loss_fn, optimizer, device, dataloaders)

if __name__ == "__main__":
    main()