'''
Descripttion: 
Result: 
Author: Philo
Date: 2023-03-10 16:50:42
LastEditors: Philo
LastEditTime: 2023-03-11 16:41:50
'''

import torch
from torch import nn
from torch.nn import functional as F

class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True) -> None:
        super().__init__()
        """
        in_channels: 输入通道
        inter_channels: 中间数据通道
        dimension: 输入数据的维度
        sub_sample: 是否进行最大池化 一般是True
        bn_layer: 一般是True
        """
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1,2,2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d
        
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)  # 使用 0 对 参数进行赋初值
            nn.init.constant_(self.W[1].bias, 0)   # 使用 0 对参数进行赋初值
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
        
    def forward(self, x):
        batch_size = x.size(0)   

        g_x = self.g(x).view(batch_size, self.inter_channels, -1) # b c w*h  这里还经过了maxpool的操作，maxpool:w_out = (w-k_size+2*pad)/k_size + 1
        print(g_x.shape, "self.g后的数据")
        g_x = g_x.permute(0, 2, 1)    #维度变化 b wh c    

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1) # b c w*h 这里没有经过maxpool操作
        print(theta_x.shape, "self.theta后的数据")
        theta_x = theta_x.permute(0, 2, 1) # b wh c

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) # b c w*h 这里经过了maxpool
        print(phi_x.shape, "self.phi_x后的数据")

        f = torch.matmul(theta_x, phi_x) # 1024*8 矩阵乘 8*256  =  1024*256 

        print(f.shape)

        f_div_C = F.softmax(f, dim=-1)  # 对 最后一维做softmax 

        y = torch.matmul(f_div_C, g_x)    # 1024*256  *   256*8   = 1024*8
        print(y.shape, "g_x和y矩阵乘后的结果")
        y = y.permute(0, 2, 1).contiguous()       # 这里的contiguous类似与clone 否则后期对y修改数据，也会对原始数据进行修改  # 得到 batch_size*8*1024
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # *x.size()[2:] 这个花里胡哨的，就是获取x的h 和 w; 再将数据恢复到原始格式
        W_y = self.W(y)  # 这里将b inter_ch w h -> b in_ch w h
        z = W_y + x  # 进行残差连接
        return z

x = torch.randn(16, 16, 32, 32)
net = NonLocalBlockND(in_channels=16)
print(net(x).shape)

