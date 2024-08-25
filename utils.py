import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def process_data(data):

    res = []
    for i in range(data.shape[0]):
        x_res = []
        for j in range(data.shape[1] - 1):
            x_res.append(str(data.iloc[i, j]))
        x_res.append(data.iloc[i, -1])
        res.append(x_res)
    return np.array(res, dtype=np.float64)

# 附用函数
def set_random_seed(state=1):

    gens = (np.random.seed, torch.manual_seed)
    for set_state in gens:
        set_state(state)

class AbsSumLoss(nn.Module):
    def __init__(self):
        super(AbsSumLoss, self).__init__()

    def forward(self, output, target):
        loss = F.l1_loss(target, output, reduction='sum')

        return loss

def train_loop(dataloader, model, loss_fn, optimizer):
    """
        模型训练
    :param dataloader: 训练数据集
    :param model: 训练用到的模型
    :param loss_fn: 评估用的损失函数
    :param optimizer: 优化器
    """
    model.train()
    for batch, x_y in enumerate(dataloader):
        X, y = x_y[:, :999].type(torch.float64), torch.tensor(x_y[:, 999], dtype=torch.long, device='cuda:0')
        with torch.set_grad_enabled(True):
            # Compute prediction and loss
            pred = model(X.float())
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            # Backpropagation
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), "CNN.pth")

def test_loop(dataloader, model, loss_fn):
    """
        模型测试部分
    :param dataloader: 测试数据集
    :param model: 测试模型
    :param loss_fn: 损失函数
    """
    size = len(dataloader.dataset)
    test_loss, correct, l1_loss = 0, 0, 0
    # 用来计算abs-sum. 等于PyTorch L1Loss
    l1loss_fn = AbsSumLoss()
    with torch.no_grad(): 
        model.eval()
        for x_y in dataloader:
            X, y = x_y[:, :999].type(torch.float64), torch.tensor(x_y[:, 999], dtype=torch.long, device='cuda:0')
            # Y用来计算L1 loss, y是CrossEntropy loss.
            Y = torch.zeros(size=(len(y), 2), device='cuda:0')  # 2为类别数
            for i in range(len(Y)):
                Y[i][y[i]] = 1
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()  
            l1_loss += l1loss_fn(pred, Y).item()  
            aaa = pred.argmax(1)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size 
    correct /= size
    print(f"Test Results:\nAccuracy: {(100 * correct):>0.1f}% abs-sum loss: {l1_loss:>8f} CroEtr loss: {test_loss:>8f}")

