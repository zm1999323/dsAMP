import h5py
import numpy as np
import pandas as pd

#
#
# def h5py_to_array(h5_path):
#     f = h5py.File(h5_path, 'r')
#     embeddings = []
#     for dataset_name in f:
#         dataset = f[dataset_name]
#         embedding = np.array(dataset)
#         embeddings.append(embedding)
#     f.close()
#
#     input_data = np.vstack(embeddings)
#
#     # print(input_data.shape)
#     # print(input_data)
#     # print(type(input_data))
#
#     return input_data
#
# # input = h5py_to_array(h5_path='/home/wuyou/project/pythonProject5/pythonProject1/protein_embeddings.h5')
# # print(input)


import torch
import torch.nn as nn
import torch.nn.functional as F


# class CNNLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes):
#         super(CNNLSTM, self).__init__()
#         self.con1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=2,bias=False)
#         self.batch1 = nn.BatchNorm1d(64)
#         self.lstm = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, num_classes) # 双向LSTM输出维度为hidden_dim*2
#
#     def forward(self, x):
#         # print("After FC:", x.shape)  # 添加调试语句，检查全连接层输出的维度
#         x = self.con1(x)
#         print("After FC1:", x.shape)
#         x = self.batch1(x)
#         print("After FC2:", x.shape)
#         x = x.permute(0,2,1)
#         x, _ = self.lstm(x)
#         x = x[:, -1, :]
#         x = self.fc(x)
#         print("After FC:", x.shape)  # 添加调试语句，检查全连接层输出的维度
#
#         return x

#
# import h5py
# import csv
#
# def h5_to_csv(h5_file, output_csv):
#     h5_datasets = {}
#     with h5py.File(h5_file, 'r') as f:
#         for key in f.keys():
#             h5_datasets[key] = f[key][:]
#
# # 将匹配的数据写入 CSV 文件
#     with open(output_csv, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Sequence Name", "Sequence"])
#         for dataset_name,dataset_data in h5_datasets.items():
#             writer.writerow([dataset_name, dataset_data])
#
#     print("Merged data saved to:", output_csv)
#
# # 使用示例
#
# h5_file = "/home/wuyou/project/pythonProject5/pythonProject1/sample/natural_cpp.h5"
# output_csv = "/home/wuyou/project/pythonProject5/pythonProject1/sample/merged_data.csv"
#
# h5_to_csv(h5_file, output_csv)







import pandas as pd
from Bio import SeqIO
import csv

# def fasta_to_csv(fasta_file, csv_file):
#     with open(csv_file, mode= 'w', newline='') as csvfile:
#         csvwriter= csv.writer(csvfile)
#
#         csvwriter.writerow(['Sequence Name' , 'Sequence Content'])
#
#         for record in SeqIO.parse(fasta_file,'fasta'):
#             sequence_name = record.id
#             sequence_content = str(record.seq)
#             csvwriter.writerow(([sequence_name, sequence_content]))
#
# fasta_file = "/home/wuyou/project/pythonProject5/pythonProject1/sample/natural_cpp.fasta"
# csv_file = "/home/wuyou/project/pythonProject5/pythonProject1/sample/natural_cpp.csv"
# fasta_to_csv(fasta_file, csv_file)



# df1 = pd.read_csv('/home/wuyou/project/pythonProject5/pythonProject1/sample/CPP_GAMP.csv')
# df2 = pd.read_csv('/home/wuyou/project/pythonProject5/pythonProject1/sample/natural_cpp.csv')
# merged_df = pd.merge(df1, df2, on='Sequence Name', how='inner')
# merged_df.to_csv('merged_file.csv', index=False)


# def merge_fasta_csv(fasta_file, csv_file, output_csv):
#     fasta_sequences = {}
#     for record in SeqIO.parse(fasta_file, "fasta"):
#         fasta_sequences[record.id] = str(record.seq)
#
# # 读取 CSV 文件，获取第一列名称
#     df = pd.read_csv(csv_file)
#     names = df.iloc[:, 0].tolist()
#
# # 匹配名称并添加序列内容到 CSV 文件
#     sequences = []
#     for name in names:
#         if name in fasta_sequences:
#             sequence = fasta_sequences[name]
#             sequences.append(sequence)
#         else:
#             sequences.append('')
#
# # 将序列内容添加到 CSV 文件的最后一列
#     df["Sequence Content"] = sequences
#
# # 保存合并后的 CSV 文件
#     df.to_csv(output_csv, index=False)
#
#     print("Merged data saved to:", output_csv)
#
#
# # 使用示例
# fasta_file = "/home/wuyou/project/pythonProject5/pythonProject1/sample/natural_cpp.fasta"
# csv_file = "/home/wuyou/project/pythonProject5/pythonProject1/sample/merged_data.csv"
# output_csv = "/home/wuyou/project/pythonProject5/pythonProject1/sample/merged_data_new.csv"
#
# merge_fasta_csv(fasta_file, csv_file, output_csv)





# import h5py
# import pandas as pd
#
# def list_h5_dataset(h5_file):
#     with h5py.File(h5_file, 'r') as f:
#         print('dataset')
#         for name in f:
#             print(name)
#
# h5_file = "/home/wuyou/project/pythonProject5/pythonProject1/sample/natural_cpp.h5"
# list_h5_dataset(h5_file)






# def h5_to_csv(h5_file, csv_file):
#     with h5py.File(h5_file, 'r') as f:
#         data = f['data'][:]
#
#     df = pd.DataFrame(data)
#
#     df.to_csv(csv_file, index=False)
#
# h5_file = "/home/wuyou/project/pythonProject5/pythonProject1/sample/natural_cpp.h5"
# csv_file = "/home/wuyou/project/pythonProject5/pythonProject1/sample/natural_cpp.csv"
#
# h5_to_csv(h5_file, csv_file)
from torch import nn
import torch
import torch.nn.functional as F


# 定义一个包含空洞卷积、批量归一化和ReLU激活函数的子模块
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            # 空洞卷积，通过调整dilation参数来捕获不同尺度的信息
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),  # 批量归一化
            nn.ReLU()  # ReLU激活函数
        ]
        super(ASPPConv, self).__init__(*modules)


# 定义一个全局平均池化后接卷积、批量归一化和ReLU的子模块
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1卷积
            nn.BatchNorm2d(out_channels),  # 批量归一化
            nn.ReLU())  # ReLU激活函数

    def forward(self, x):
        size = x.shape[-2:]  # 保存输入特征图的空间维度
        x = super(ASPPPooling, self).forward(x)
        # 通过双线性插值将特征图大小调整回原始输入大小
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


# ASPP模块主体，结合不同膨胀率的空洞卷积和全局平均池化
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256  # 输出通道数
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1卷积用于降维
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 根据不同的膨胀率添加空洞卷积模块
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 添加全局平均池化模块
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 将所有模块的输出融合后的投影层
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),  # 融合特征后降维
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))  # 防止过拟合的Dropout层

    def forward(self, x):
        res = []
        # 对每个模块的输出进行收集
        for conv in self.convs:
            res.append(conv(x))
        # 将收集到的特征在通道维度上拼接
        res = torch.cat(res, dim=1)
        # 对拼接后的特征进行处理
        return self.project(res)


# 示例使用ASPP模块
aspp = ASPP(256, [6, 12, 18])
x = torch.rand(2, 256, 13, 13)
print(aspp(x).shape)  # 输出处理后的特征图维度
