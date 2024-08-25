import torch  
import torchvision.transforms as transforms 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from net import Model  
  
def load_model_and_weights(model_path):  
    """  
        加载模型架构和预训练权重  
    :param model_path: 权重文件的路径  
    :return: 加载了权重的模型实例  
    """  
    model = Model()  # 创建模型实例  
    model.load_state_dict(torch.load(model_path))  # 加载预训练权重  
    model.eval()  # 将模型设置为评估模式  
    if torch.cuda.is_available():  
        model.cuda()  # 如果GPU可用，则将模型移动到GPU  
    return model  
  
def predict(model, data):  
    """  
    使用加载了权重的模型进行预测  
    :param model: 加载了权重的模型实例  
    :param data: 待预测的数据,假设为torch.Tensor,并且已经是正确的格式和类型  
    :return: 预测结果  
    """  
    # 如果模型在GPU上，确保数据也在GPU上  
    if next(model.parameters()).is_cuda:  
        data = data.cuda()  
      
    # 进行预测，不需要计算梯度  
    with torch.no_grad():  
        predictions = model(data.float())  
      
    # 假设预测是概率分布，取概率最大的类别的索引作为预测结果  
    _, predicted_indices = torch.max(predictions, 1)  
      
    return predicted_indices  

def get_pred_x(data):

    res = []
    for i in range(data.shape[0]):
        x_res = []
        for j in range(data.shape[1] - 1):
            x_res.append(str(data.iloc[i, j]))
        res.append(x_res)
    return np.array(res, dtype=np.float64)

# 加载模型和权重  
model = load_model_and_weights('CNN.pth')  

b_size = 1
pred_data = pd.read_csv('test.csv')
pred_data = get_pred_x(pred_data)

pred_data = torch.cuda.FloatTensor(pred_data)
pre_loader = torch.utils.data.DataLoader(dataset=pred_data, batch_size=b_size, shuffle=True)

# 遍历数据集进行预测  
for batch, x in enumerate(pre_loader):  
    # 假设x已经是正确的格式和类型，并且如果模型在GPU上，x也已经在GPU上  
    predictions = predict(model, x[:, :999])  # 只取特征部分进行预测  
    print(predictions)
