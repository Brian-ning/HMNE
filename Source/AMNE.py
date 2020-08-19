import os
import torch
import torch.nn as nn
import Reader
import Graph2Coo
import GATModel
import numpy as np
import pickle as pk
import Evaluation as eval
import cross_layer_walk as clw

transform = {
    '1':lambda a, b, c: a + (b + c),
    '2':lambda a, b, c: torch.mm(torch.mm(b, c.t()), a),
    '3':lambda a, b, c: torch.mul(a, b*c)
}

class train_model:
    def __init__(self, path, sampling, dimension):
        self.path = path
        self.sampling = sampling
        self.dimension = dimension
        self.data = None

    def train_AMNE(self):
        # 加载数据集
        dataset, edges_list, edges_label, nodes_mat = self.dataset_load() # 加载数据集，测试边集和相应的标签
        number_graphs = len(dataset) # 图的个数
        model = GATModel.Net(dataset, number_graphs) # 初始化模型
        # 设置目标函数和优化方法
        criterion = nn.BCELoss() #要是解决多分类任务的目标函数
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001, lr=0.0001) # 设置优化器

        pre_acc = 0 # 用来保存最优的模型情况
        for epoch in range(1, 3001):
            optimizer.zero_grad()
            fuse_feat, used_feat, comp_re, obf1, obf0 = model() # 前向计算,返回的这些值应该不需要计算梯度，除了used_feat

            obf2 = criterion(fuse_feat, torch.tensor(dataset[0].x, dtype=torch.float32)) # 初始的属性与融合之后的属性
            obf3 = criterion(comp_re, nodes_mat) # 互补性信息的计算
            object_function = obf0 + obf1 + obf2 + obf3 # 总的目标函数

            # 测试集上验证学习到的嵌入的性能
            Acc = []
            Adj = []
            # 循环10次，求平均值
            for i in range(10):
                # 利用5交叉验证,计算得到准确性和另外一种指标
                accuracy, adjust = eval.link_prediction(used_feat, edges_list, edges_label, dimensions = 128, GCN=True)
                Acc.append(accuracy)
                Adj.append(adjust)
            # 求平均值
            average_acc = sum(Acc)/10
            average_adj = sum(Adj)/10

            print("----Epoch: %d -----Loss = %.4f : %.4f/ %.4f/ %.4f/ %.4f ----Accuracy = %.4f ----ADJ Score = %.4f"%(epoch, object_function, obf0, obf1, obf2, obf3, average_acc, average_adj))            # 最优模型的输出和保存
            if pre_acc < average_acc:
                max_accuracy = average_acc
                max_adjust = average_adj
                pre_acc = average_acc
                Max_acc = Acc
                Max_adj = Adj
                torch.save(model.state_dict(),'./model/model.pt')
            # 模型的反向传播和参数修改
            object_function.backward() # 反向传播，更新梯度
            optimizer.step()
        print(" ----Max Accuracy : %.4f ---- MAx ADJ Score: %.4f ----"%(max_accuracy, max_adjust))
        print([(Max_acc[i], Max_adj[i]) for i in range(len(Max_acc))])

    def dataset_load(self):
        path = "baselines.pkl"
        if os.path.exists(path):
            print("The pkl file has existed!")
            with open(path, 'rb') as f:
                mul_nets, _, pos_edge_list, neg_edge_list, nodes_attr = pk.load(f)
        else:
            file_path = "./Sampling_graph/Datasets_With_Attributes/"+ os.path.basename(self.path) + ".graph"
            mul_nets, _, pos_edge_list, neg_edge_list, nodes_attr = Reader.data_load(file_path)
        nodes_prob = clw.RWGraph(mul_nets) # 加载节点局部信息互补性的采样度量
        nodes_mat = np.zeros((mul_nets[0].number_of_nodes(), len(mul_nets)*len(mul_nets)))
        for n in sorted(list(mul_nets[0].nodes())):
            nodes_mat[int(n),:] = nodes_prob.layer_transition_prob[n].flatten()

        graph_list = Graph2Coo.graphs2coo(mul_nets, nodes_attr)
        dataset = Graph2Coo.CreatMyDataset(graph_list, '../Benchmark/')
        for i in range(len(mul_nets)):
            dataset[i].edge_index = torch.tensor(dataset[i].edge_index, dtype=torch.long)
        edges_list, labels = get_selected_edges(pos_edge_list, neg_edge_list)
        return dataset, edges_list, labels, torch.from_numpy(nodes_mat).to(torch.float32)

    def test_modal(self):
        path = './model/model.pt'
        dataset, edges_list, edges_label, nodes_attr = self.dataset_load()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists(path):
            model = GATModel.Net(dataset, len(dataset))
            model.load_state_dict(torch.load(path))
            _, used_feat, _, _, _ = model.forward()
            for i in range(100):
                accuracy, adjust = eval.link_prediction(used_feat, edges_list, edges_label,  dimensions = 128, GCN=True)
                print(" ---- Accuracy : %.4f ---- ADJ Score: %.4f ----"%(accuracy, adjust))
        else:
            print('The model has saved in this file path!')

def get_selected_edges(pos_edge_list, neg_edge_list):
    edges = pos_edge_list + neg_edge_list
    labels = np.zeros(len(edges))
    labels[:len(pos_edge_list)] = 1
    return edges, labels
