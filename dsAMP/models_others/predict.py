import os
import json
import torch
from setnet2 import Classifier
import csv
from utils import h5py_to_tensor

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predict_path = '/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/data/AMP.h5'
    # 获取嵌入数据
    data = h5py_to_tensor(h5_path=predict_path)
    # 扩展 batch 维度
    data = torch.unsqueeze(data, dim=0)
    data = data.permute(1,2,0)

    # 读取类别索引映射
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), "File: '{}' does not exist.".format(json_path)
    #
    # with open(json_path, "r") as f:
    #     class_indict = json.load(f)

    # 创建模型
    net = Classifier(num_classes=2).to(device)

    # 加载模型权重
    weights_path = "/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/best_model_fold_9.pth"
    # assert os.path.exists(weights_path), "File: '{}' does not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path))

    net.eval()
    with torch.no_grad():
        # 预测类别
        output = net(data.to(device)).cpu()
        print(output)
        predict_cla = torch.argmax(output, dim=1)

    # class_indict = {0: 'non-G-AMP', 1:'G-AMP'}
    # csv_path = "/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/best_model_fold_1neg.csv"
    # with open(csv_path, mode='w', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=["Class", "Probability"])
    #     writer.writeheader()
    #     for i in range(len(output)):
    #         writer.writerow({"Class": class_indict[predict_cla[i].item()], "Probability": output[i,predict_cla[i]].item()})

    # print("Results saved to:", csv_path)

    predict = torch.max(output, dim=1)[1]
    print(predict)
    num_ones = torch.eq(predict, 1).sum().item()
    print(num_ones)
    num_zeros = torch.eq(predict, 0).sum().item()
    print(num_zeros)


if __name__ == '__main__':
    main()

