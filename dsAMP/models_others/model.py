import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CNNLSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=1, stride=1,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.2),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, stride=1,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            # nn.Dropout(0.2),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
            )
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        # self.lstm2 = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        # self.drop = nn.Dropout(0.5)
        # self.batch = nn.BatchNorm1d(hidden_dim * 2)
        # self.fc = nn.Linear(hidden_dim * 2, 16)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes)# 双向LSTM输出维度为hidden_dim*2
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.permute(0, 2, 1) 的作用是将维度从 [batch, features, sequence_length] 调整为 [batch, sequence_length, features]
        # x = x.permute(0,1,2)
        # print( x.shape)  # 添加调试语句，检查全连接层输出的维度
        # 0: batch, 2: sequence_length, 1: features
        # 输入 Conv1d 层，输出维度为 [batch, out_channels, new_sequence_length]
        x = self.conv(x)
        # print("After conv:", x.shape)
        # 输入 LSTM 层，x 维度为 [batch, sequence_length, input_size]
        # LSTM 需要的输入维度为 [batch, sequence_length, hidden_size]
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        # x, _ = self.lstm2(x)
        # print("After LSTM:", x.shape)  # 添加调试语句，检查 LSTM 输出的维度
        # 取 LSTM 输出的最后一个时间步作为整个序列的表示
        x = x[:, -1, :]
        # x = x.view(x.size(0), -1)
        # x = self.drop(x)
        # x = self.batch(x)
        # print("After taking last timestep:", x.shape)  # 添加调试语句，检查取最后一个时间步后的维度
        # x = self.fc(x)
        # x = self.relu(x)
        # x = self.drop(x)
        x = self.sigmoid(self.fc2(x))
        # print("After FC:", x.shape)  # 添加调试语句，检查全连接层输出的维度

        return x




# model = CNNLSTM(input_dim=1024, hidden_dim=64, num_classes=2)
# batch_size = 32
#
# input_tensor = torch.randn(batch_size,1024)
# input_tensor = input_tensor.unsqueeze(2)
# out_tensor = model(input_tensor)







# model = CNNLSTM()
# batch_size = 32
#
# input_tensor = torch.randn(batch_size,1024)
# input_tensor = input_tensor.unsqueeze(2)
# out_tensor = model(input_tensor)


# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torch.nn.utils.rnn import pack_padded_sequence
# import torch.nn.functional as F
# from torchvision.models import resnet18, resnet101
#
#
# class CNNLSTM(nn.Module):
#     def __init__(self, num_classes=2):
#         super(CNNLSTM, self).__init__()
#         self.resnet = resnet101()
#         self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
#         self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
#         self.fc1 = nn.Linear(256, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         x = self.resnet(x)
#         x = x.unsqueeze(1)
#         out, _ = self.lstm(x)
#         x = self.fc1(out[:, -1, :])
#         x = F.relu(x)
#         x = self.fc2(x)
#         return x
#
#
# model = CNNLSTM()
# batch_size = 32
#
# input_tensor = torch.randn(batch_size,1024)
# input_tensor = input_tensor.unsqueeze(2)
# out_tensor = model(input_tensor)