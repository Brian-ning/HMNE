import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import copy



class Net(torch.nn.Module):
    def __init__(self, dataset, device):
        super(Net, self).__init__()
        self.device = device
        self.dataset = dataset # 数据集加载
        # 线性特征提取
        self.fc1 = nn.Linear(dataset.num_features, 128) # 节点特征MLP提取
        # 卷积操作
        self.conv1 = GCNConv(128, 128)
        self.conv2 = GCNConv(128, 128)
        # 反卷积操作
        self.dconv1 = GCNConv(128, 128)
        self.dconv2 = GCNConv(128, 128)
        # 线性融合, 节点特征的降维融合
        self.fc2 = nn.Linear(128, dataset.num_features)

        # 反向传播更新的参数
        self.w1 = self.fc1.parameters()
        self.conv1_reg_params1 = self.conv1.parameters()
        self.conv2_reg_params1 = self.conv2.parameters()
        self.dconv1_reg_params2 = self.dconv1.parameters()
        self.dconv2_reg_params2 = self.dconv2.parameters()
        self.w2 = self.fc2.parameters()

    def forward(self):
        pre_x = [] # 关键属性的提取
        normal_x = [] # 每层中节点的分别嵌入
        # 图数据中每个数据集分别卷积
        for data in self.dataset:
            x, edge_index = data.x, data.edge_index
            x = self.fc1(x)
            pre_x.append(x)
            x = self.conv1(x, edge_index)
            x = self.conv2(x, edge_index)
            normal_x.append(x)

        # 初始化变量
        graph_number = len(normal_x)
        graph_id = list(range(graph_number))
        # 不同层中节点嵌入的混合
        xs = []
        for i in graph_id:
            edge_index = self.dataset[i].edge_index
            sum_x = torch.zeros(normal_x[i].size())
            for j in graph_id:
                if i != j:
                    sum_x += normal_x[j]
            # 反卷积操作，实际公式和真实的卷积相同
            normal_x[i] = self.dconv1(sum_x, edge_index)
            normal_x[i] = self.dconv2(normal_x[i], edge_index)
            xs.append(normal_x[i])

        # 节点的多特征融合部分需要进一步论证，需要一种比较好的方法
        fin_feat = xs[0]
        for feat_index in range(1, len(xs)-1):
            fin_feat = fin_feat * xs[feat_index]
        # fin_feat = F.log_softmax(fin_feat)
        fin_feat = F.sigmoid(fin_feat)
        Loss_embedding = self.fc2(fin_feat)
        return pre_x, xs, fin_feat, Loss_embedding