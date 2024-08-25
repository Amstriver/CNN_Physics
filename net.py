import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),  # 16, 1, 999
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=4),                                 # 32, 1, 250
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 64, 1, 250
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.conv_layer4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=5, stride=5),                                 # 128, 1, 50
        )

        self.conv_layer5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 256, 1, 50
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.conv_layer6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=5, stride=5),                                  # 512, 1, 10
        )

        self.full_layer = nn.Sequential(
            nn.Linear(in_features=512 * 10, out_features=512 * 10),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=512 * 10, out_features=512 * 10),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=512 * 10, out_features=2)
        )

        self.pred_layer = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = x.view(x.size(0), -1)
        x = self.full_layer(x)

        if self.training:
            return x
        else:
            return self.pred_layer(x)