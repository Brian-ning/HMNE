import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot


class Net(torch.nn.Module):
    def __init__(self, dataset, number_graphs):
        super(Net, self).__init__()
        self.number_graphs = number_graphs
        self.dataset = dataset # 数据集加载
        # 线性特征提取
        self.fc1 = nn.ModuleList([nn.Linear(self.dataset.num_features, self.dataset.num_features) for _ in range(self.number_graphs)])
        # 卷积操作
        self.conv1 = GCNConv(self.dataset.num_features, 128)
        self.conv2 = GCNConv(128, 128)
        # self.conv3 = GCNConv(128, 128)
        # 反卷积操作
        self.deconv1 = GCNConv(128, 128)
        self.deconv2 = GCNConv(128, self.dataset.num_features)
        self.ReLU = nn.ReLU()
        # 线性融合, 节点特征的降维融合
        self.fc2 = nn.Linear(2*self.dataset.num_features, self.dataset.num_features)
        self.fc3 = nn.Linear(self.dataset.num_features, 3)

        self.edge_index = [torch.tensor(data.edge_index, dtype=torch.long) for data in self.dataset]
        self.weight = Parameter(torch.Tensor(self.number_graphs, 128, 128))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self):
        # 每层进行特征提取
        pre_feat = []  # 初始编码
        conv1_f = []   # 第一级编码
        encoder_H = [] # 第二级编码，也是最后使用的节点嵌入
        decov1_f = []  # 第一级解码
        h_1_all = []   # 第三级解码，也是最终用于融合特征形成

        for i in range(self.number_graphs):
            h_1 = self.fc1[i](torch.tensor(self.dataset[i].x, dtype=torch.float32)) # 对于原始特征的提取操作, 这个步骤是有效的
            pre_feat.append(h_1)
            # 非共享参数的编码计算，目的是破除论文中的一些假设
            h_1 = self.conv1(h_1, self.edge_index[i])
            conv1_f.append(h_1)
            h_1 =  self.conv2(h_1, self.edge_index[i])
            encoder_H.append(h_1) # 用于存储编码阶段的表示，主要受启发ATE，点云+边卷积等算法的启发
            h_1_all.append(h_1) # 用于下一步的解码计算

        # read_out 操作得到每层图的嵌入
        global_vec = []
        for i in range(self.number_graphs):
            global_vec.append(F.sigmoid(sum(encoder_H[i])/encoder_H[i].shape[0]))

        # 不同层中节点嵌入的混合
        decoder_h = []
        each_fin_feat = []
        sum_decoder = sum(h_1_all)
        for i in range(self.number_graphs):# 计算每个网络卷积之后的综合
            decoder_h.append(torch.div((sum_decoder - h_1_all[i]), (self.number_graphs-1))) # 求其他层的节点嵌入平均

        # 解码阶段
        for i in range(self.number_graphs):
            edge_index = self.edge_index[i]
            # 反卷积操作，实际公式和真实的卷积相同
            h_1_all[i] = self.deconv1(h_1_all[i], edge_index)
            decov1_f.append(h_1_all[i])
            h_1_all[i] = self.deconv2(h_1_all[i], edge_index)
            # 共享参数的解码阶段
            decoder_h[i] = self.deconv1(decoder_h[i], edge_index)
            decoder_h[i] = self.deconv2(decoder_h[i], edge_index)
            fuse_h = torch.cat((h_1_all[i], decoder_h[i]), 1) #
            each_fin_feat.append(self.fc2(fuse_h)) # 每层的嵌入表示，维度为13

        comp_label = []
        for i in range(self.number_graphs):
            comp_inf = []
            for j in range(self.number_graphs):
                g_l_em = F.sigmoid(torch.matmul(torch.matmul(encoder_H[j], self.weight[i,:]), global_vec[i].unsqueeze(dim=0).t()))
                comp_inf.append(g_l_em)
            layer_i_all = torch.cat((comp_inf), 1)
            layer_i_all = layer_i_all * torch.div(1, torch.sum(layer_i_all, 1, keepdim=True).repeat((1,self.number_graphs)))
            comp_label.append(layer_i_all)
        comp_re = torch.cat((comp_label), 1) # 最后生成的节点互补性信息矩阵，也可以看作节点在不同层中的相似性

        # 节点的多特征融合部分需要进一步论证，需要一种比较好的方法
        fuse_feat = F.sigmoid(sum(each_fin_feat) / self.number_graphs)
        used_feat = F.elu(sum(encoder_H) / self.number_graphs)

        obf0 = 0
        obf1 = 0 # 第二种目标函数, 设置为卷积-反卷积的输入和输出是相似的
        # 每一层网络都需要计算目标函数
        for i in range(self.number_graphs):
            # en_de_obj = torch.norm(pre_feat[i]-h_1_all[i], p=2) + torch.norm(encoder_H[i]-decov1_f[i], p=2) + torch.norm(conv1_f[i]-decov2_f[i], p=2)
            en_de_obj = torch.norm(pre_feat[i]-h_1_all[i], p=2) + torch.norm(conv1_f[i]-decov1_f[i], p=2)
            obf1 = obf1 + torch.div(en_de_obj,2)
            for j in range(self.number_graphs):
                if i!=j:
                    obf0 = obf0 + torch.norm(each_fin_feat[i]-each_fin_feat[j], p=2)

        return fuse_feat, used_feat, comp_re, obf1, obf0