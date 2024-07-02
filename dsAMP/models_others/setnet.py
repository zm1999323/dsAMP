import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
class SEAttention(nn.Module):
    # 初始化SE模块，channel为通道数，reduction为降维比率
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，将特征图的空间维度压缩为1x1
        self.fc = nn.Sequential(  # 定义两个全连接层作为激励操作，通过降维和升维调整通道重要性
            nn.Linear(channel, channel // reduction, bias=False), # 降维，减少参数数量和计算量
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),# ReLU激活函数，引入非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 升维，恢复到原始通道数
            nn.Sigmoid(),
            nn.Dropout(0.6)# Sigmoid激活函数，输出每个通道的重要性系数
        )

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入x的批量大小b和通道数c
        y = self.avg_pool(x).view(b, c)
        # print(y.shape)# 通过自适应平均池化层后，调整形状以匹配全连接层的输入
        y = self.fc(y).view(b, c , 1, 1)
        # print(y.shape)# 通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
        return y  # 将通道重要性系数应用到原始特征图上，进行特征重新校准



def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保减小的百分比不超过一定的比例（round_limit）
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

# Radix Softmax用于处理分组特征的归一化
class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        # 根据radix是否大于1来决定使用softmax还是sigmoid进行归一化
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = x.sigmoid()
        return x

# SplitAttn模块实现分裂注意力机制
class SplitAttn(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
                 act_layer=nn.LeakyReLU, norm_layer=None, drop_block=None, **kwargs):
        super(SplitAttn, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix
        # 根据输入通道数、radix和rd_ratio计算注意力机制的中间层通道数
        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        # 核心卷积层
        self.conv = nn.Conv2d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)
        # 后续层以及RadixSoftmax
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.act0 = act_layer()
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)
        # self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        # 卷积和激活
        x = self.conv(x)
        x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        # 计算分裂注意力
        B, RC, H, W = x.shape
        if self.radix > 1:
            # 对特征进行重组和聚合
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        # 全局平均池化和两层全连接网络，应用RadixSoftmax
        x_gap = x_gap.mean(2, keepdims=True).mean(3, keepdims=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        # x_gap = self.dropout(x_gap)
        x_attn = self.fc2(x_gap)
        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out



# # 输入 N C H W,  输出 N C H W
# if __name__ == '__main__':
#     block = SplitAttn(64)
#     input = torch.rand(1, 64, 64, 64)
#     output = block(input)
#     print(output.shape)



class Classifier(nn.Module):
    def __init__(self, num_classes=2,hidden_dim=128):
        super(Classifier, self).__init__()
        self.feartures = nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(512),
            SEAttention(512,reduction=16),
            nn.Conv2d(512,256,kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            SplitAttn(256),
            nn.Conv2d(256,128,kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        # self.lstm = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(128 , 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes))
        # 假设输入特征大小为 512

    def forward(self, x):
        x = x.unsqueeze(0)
        # print(x.shape)
        x = x.permute(1,2,0,3)
        # print(x.shape)
        x = self.feartures(x)
        # x = self.batch(x)
        # print(x.shape)
        # x = self.avg_pool(x)
        x = torch.squeeze(x,dim=3)
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # x, _ = self.lstm(x)
        # print(x.shape)
        # x = self.avg_pool(x)
        x, _ = self.lstm(x)
        # print("After LSTM:", x.shape)  # 添加调试语句，检查 LSTM 输出的维度
        # 取 LSTM 输出的最后一个时间步作为整个序列的表示
        x = x[:, -1, :]
        # x = torch.flatten(x, 1)
        # x = self.drop(x)
        x = self.fc(x)
        return x

# if __name__ == '__main__':
#     input = torch.randn(64, 1024, 1)  # 随机生成一个输入特征图
#     se = Classifier()  # 实例化SE模块，设置降维比率为8
#     output = se(input)  # 将输入特征图通过SE模块进行处理
#     print(output.shape)  # 打印处理后的特征图形状，验证SE模块的作用





