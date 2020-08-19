import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import copy
import numpy as np



class Net(torch.nn.Module):
    def __init__(self, dataset, number_graphs):
        super(Net, self).__init__()
        self.number_graphs = number_graphs
        self.dataset = dataset # 数据集加载
        # 线性特征提取
        self.fc1 = nn.ModuleList([nn.Linear(dataset.num_features, 128) for _ in range(self.number_graphs)])
        # 卷积操作
        self.conv1 = nn.ModuleList([GCNConv(128, 128) for _ in range(self.number_graphs)])
        self.conv2 = nn.ModuleList([GCNConv(128, 128) for _ in range(self.number_graphs)])
        # self.ReLU1 = nn.ModuleList([nn.ReLU(True) for _ in range(self.number_graphs)])
        # 反卷积操作
        self.dconv1 = nn.ModuleList([GCNConv(128, 128) for _ in range(self.number_graphs)])
        self.dconv2 = nn.ModuleList([GCNConv(128, 128) for _ in range(self.number_graphs)])
        self.ReLU = nn.ReLU()
        # 线性融合, 节点特征的降维融合
        self.fc2 = nn.Linear(128, dataset.num_features)

    def forward(self):
        # 每层进行特征提取
        h_1_all = []
        encoder_H = []
        pre_feat = []
        for i in range(self.number_graphs):
            h_1 = self.fc1[i](self.dataset[i].x)
            pre_feat.append(h_1)
            h_1 = self.conv1[i](h_1, self.dataset[i].edge_index)
            h_1 = self.conv2[i](h_1, self.dataset[i].edge_index)
            h_1 = self.ReLU(h_1)
            h_1_all.append(h_1)
            encoder_H.append(h_1)

        # 不同层中节点嵌入的混合
        x_gs = []
        for i in range(self.number_graphs):
            edge_index = self.dataset[i].edge_index
            for j in range(self.number_graphs):
                if j != i:
                    h_1_all[i] = h_1_all[i] + h_1_all[j]
            h_1_all[i] = h_1_all[j] - h_1_all[i]
            # 反卷积操作，实际公式和真实的卷积相同
            h_1_all[i] = self.dconv1[i](h_1_all[i], edge_index)
            h_1_all[i] = self.dconv2[i](h_1_all[i], edge_index)
            # h_1_all[i] = self.ReLU(h_1_all[i])
        # 节点的多特征融合部分需要进一步论证，需要一种比较好的方法
        fin_feat = h_1_all[0]
        for feat_index in range(1, self.number_graphs-1):
            fin_feat = fin_feat + h_1_all[feat_index]
        # fin_feat = fin_feat
        Loss_embedding = F.softmax(self.fc2(fin_feat))
        return pre_feat, encoder_H, h_1_all, fin_feat, Loss_embedding