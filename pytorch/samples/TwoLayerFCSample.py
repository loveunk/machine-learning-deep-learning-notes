import torch

N = 100
lr = 1e-3
epochs = 1000

D_in, H, D_out = 100, 500, 100
x = torch.randn(N, D_in).cuda()
y = torch.randn(N, D_out).cuda()

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.l1 = torch.nn.Linear(D_in, H)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

#torch.nn.init.normal_(model[0].weight)
#torch.nn.init.normal_(model[2].weight)
model = TwoLayerNet(D_in, H, D_out).cuda()

loss_fn = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for it in range(epochs):
    # forward pass
    y_pred = model(x).cuda()

    # calc loss
    loss = loss_fn(y_pred, y)
    print(it, loss.cpu().item())

    optimizer.zero_grad()

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()

