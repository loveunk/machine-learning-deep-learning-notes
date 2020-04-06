import torch

X = torch.tensor([[0.],[1.],[2.],[3.]])
Y = torch.tensor([[0.],[1.],[2.],[3.]])

w = torch.tensor([[0.0]], requires_grad=True)
b = torch.tensor([[0.0]], requires_grad=True)

for i in range(1000):
    preds = torch.mm(X, w)
    preds = torch.add(preds, b)
    preds = torch.nn.functional.relu(preds)

    # loss
    loss = torch.nn.functional.mse_loss(preds, Y)
    print('epoch: {}, loss: {}'.format(i, loss.item()))

    loss.backward()

    # grad
    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad

        w.grad.zero_()
        b.grad.zero_()

print('w: {}, b: {}'.format(w.item(), b.item()))
