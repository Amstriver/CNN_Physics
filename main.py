import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import time

from sklearn.model_selection import train_test_split

from net import Model  
from utils import *


b_size = 1
data = pd.read_csv('train.csv')
data = process_data(data)

# 拆分训练测试集
train, test = train_test_split(data, test_size=0.2)
# 将DataFrame转换为NumPy数组  
# train_np = train.values  
# test_np = test.values  

# # 将NumPy数组转换为PyTorch张量，并确保数据类型为float32  
# train_tensor = np.array(train_np, dtype=np.float64)
# test_tensor = np.array(test_np, dtype=np.float64)
train, test = torch.cuda.FloatTensor(train), torch.cuda.FloatTensor(test)
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=b_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=b_size)

# 初始化模型
adm_lr = 1e-4
sgd_lr = 5e-6
mom = 0.8
w_decay = 1e-5
n_epoch = 300
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Model()
net.to(device)
optimizer = torch.optim.Adam(params=net.parameters(), lr=adm_lr, weight_decay=w_decay, amsgrad=True)
loss_fn = nn.CrossEntropyLoss(reduction='sum')

for epoch in range(n_epoch):
    if epoch == 20:
        optimizer = torch.optim.SGD(params=net.parameters(), lr=sgd_lr, weight_decay=w_decay, momentum=mom)
    start = time.time()
    print(f"\n----------Epoch {epoch + 1}----------")
    train_loop(train_loader, net, loss_fn, optimizer)
    test_loop(test_loader, net, loss_fn)
    end = time.time()
    print('training time: ', end - start)