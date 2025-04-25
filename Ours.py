import torch.optim as optim
import torch


# 标签转onehot编码
def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot


# 调整图像对比度
def apply_contrast(images, contrasts=[1, 1.3, 1.6, 2]):
    bs, _, H, W = images.shape
    outputs = []

    for contrast in contrasts:
        transformed = (images - images.mean(dim=(2, 3), keepdim=True)) * contrast + images.mean(dim=(2, 3), keepdim=True)
        outputs.append(transformed)     #把图片的对比度调为 contrast

    return torch.cat(outputs, dim=1)


import torch
import torch.nn as nn


# 独立通道卷积模块
class IndependentChannelConv(nn.Module):
    def __init__(self, in_channels, out_channels_per_channel=3, kernel_size=3, padding=1):
        super(IndependentChannelConv, self).__init__()
        self.out_channels_per_channel = out_channels_per_channel  # 每个通道独立产生 3 个通道
        # 创建 C 个独立的卷积层，每个只作用于单通道
        self.convs = nn.ModuleList([
            nn.Conv2d(1, out_channels_per_channel, kernel_size=kernel_size, padding=padding)
            for _ in range(in_channels)
        ])

    def forward(self, x):
        x_split = torch.split(x, 1, dim=1)  # 输入张量在通道维度拆分成 C 个 [bs, 1, H, W]

        out = [torch.nn.functional.mish(conv(xi)) for conv, xi in zip(self.convs, x_split)]  # 每个通道独立应用卷积
        return torch.cat(out, dim=1)  # 在通道维度拼接，得到 [bs, 3C, H, W]



import torch.nn.functional as F

import torch.nn as nn
import torchvision

# 带注意力机制的CNN的model
class UltraAttentionCNN(nn.Module):
    def __init__(self, baseline, in_channels=4):
        super(UltraAttentionCNN, self).__init__()

        self.base_model = baseline

        # 修改第一个卷积层的输入通道数，将3改为16
        self.base_model.conv1 = nn.Conv2d(in_channels*3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 修改最后的全连接层的输出类别数，将1000改为num_classes
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 4)

        self.depthwise_conv = IndependentChannelConv(in_channels)

        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, in_channels)  # Reduce dimension
        self.fc2 = nn.Linear(in_channels, in_channels)

    def forward(self, x):

        x = apply_contrast(x)  #bs, grid, H, W。 其中的grid是不同设置的Contrast的图片的数量

        x_conv = self.depthwise_conv(x) #bs, grid*3, H, W
        bs,_,H,W = x_conv.shape

        x_pool = x_conv.view(bs, self.in_channels, 3, H, W).mean(dim=(2, 3, 4))
        attention = F.relu(self.fc1(x_pool))
        attention = torch.sigmoid(self.fc2(attention))  # Sigmoid to get attention weights in range [0, 1]

        attention_expanded = attention.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [bs, grid, 1, 1, 1]
        attention_expanded = attention_expanded.repeat(1, 1, 3, H, W)  # [bs, grid, 3, H, W]

        # 调整输入图像形状为 [bs, grid, 3, H, W] 以匹配注意力
        input_tensor_reshaped = x_conv.view(bs, self.in_channels, 3, H, W)
        # 逐通道加权
        weighted_tensor = input_tensor_reshaped * attention_expanded  # 逐元素相乘

        # 变回原来的形状 [bs, grid*3, H, W]
        out = weighted_tensor.view(bs, self.in_channels * 3, H, W)

        # out = self.base_model(out)
        # return out

        out = self.base_model(out)
        return out


#获取使用ResNet18作为基础模型的UltraAttentionCNN
def Get_BG_ResNet18():
    return UltraAttentionCNN(torchvision.models.resnet18(pretrained=False))

#获取使用ResNet34作为基础模型的UltraAttentionCNN
def Get_BG_ResNet34():
    return UltraAttentionCNN(torchvision.models.resnet34(pretrained=False))