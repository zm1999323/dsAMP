import torch
from torch.utils.data import Dataset, DataLoader
import json

class ProteinDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_sample = self.X[idx]
        X_sample = torch.tensor(X_sample)
        X_sample = X_sample.unsqueeze(-1)
        y_sample = self.y[idx]
        y_sample = torch.tensor(y_sample, dtype=torch.long)  # 转换标签为 Tensor 对象
        return X_sample, y_sample

    def get_targets(self):
        return self.y



AMP_list = {'AntiGarm_negative':0, 'AntiGarm_positive':1}
cla_dict = dict((val, key) for key, val in AMP_list.items())
    # write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)


#
# import pandas as pd
#
# def excel_to_fasta(excel_file, sheet_name, sequence_column, output_file):
#     # 读取Excel文件
#     df = pd.read_excel(excel_file, sheet_name=sheet_name)
#
#     # 获取氨基酸序列列
#     sequences = df[sequence_column].tolist()
#
#     # 创建FASTA格式字符串
#     fasta_lines = []
#     for i, seq in enumerate(sequences, start=1):
#         fasta_lines.append(f">Sequence_{i}")
#         fasta_lines.append(seq)
#
#     # 将FASTA格式保存到文件中
#     with open(output_file, 'w') as f:
#         f.write('\n'.join(fasta_lines))
#
# # 示例调用
# excel_file = 'path/to/your/excel_file.xlsx'
# sheet_name = 'Sheet1'  # Excel表格的工作表名
# sequence_column = 'Amino Acid Sequence'  # 包含氨基酸序列的列名
# output_file = 'output.fasta'  # 输出的FASTA文件路径
#
# excel_to_fasta(excel_file, sheet_name, sequence_column, output_file)

import h5py
import numpy as np
import pandas as pd



def h5py_to_tensor(h5_path):
    f = h5py.File(h5_path, 'r')
    embeddings = []
    for dataset_name in f:
        dataset = f[dataset_name]
        embedding = np.array(dataset)
        embeddings.append(embedding)
    f.close()

    input_data = np.vstack(embeddings)
    input_data = torch.tensor(input_data)

    # print(input_data.shape)
    # print(input_data)
    # print(type(input_data))

    return input_data

# input = h5py_to_array(h5_path='/home/wuyou/project/pythonProject5/pythonProject1/protein_embeddings.h5')
# print(len(input))
# print(input.shape)
# print(input)
# print(type(input))