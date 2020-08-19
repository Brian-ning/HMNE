import os
import torch
import torch.nn as nn
import Reader
import Graph2Coo
import GATModel
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
import Evaluation as eval

class train_model:
    def __init__(self, path, sampling, dimension):
        self.path = path
        self.sampling = sampling
        self.dimension = dimension
        self.data = None

    def train_AMNE(self):
        # 加载数据集
        dataset, data_merge, edges_list, edges_label, nodes_attr = self.data_load()
        number_graphs = len(dataset)
        model = GATModel.Net(dataset, number_graphs)

        # 设置目标函数和优化方法
        criterion2 = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01, lr=0.001, betas=(0.9,0.999))

        pre_acc = 0
        for epoch in range(1, 6001):
            pre_feat, encoder_H, decoder_H, fin_feat, feat_orig = model()
            optimizer.zero_grad()
            object_function1 = 0
            for i in range(number_graphs):
                object_function1 = object_function1 + torch.norm(pre_feat[i]-decoder_H[i], p=2)#torch.norm(encoder_H[i]-decoder_H[i], p=2) #+
            object_function = object_function1 + torch.norm(feat_orig - dataset[0].x, p=2)
            object_function.backward()
            optimizer.step()
            Acc = []
            Adj = []
            for i in range(10):
                accuracy, adjust = eval.link_prediction(fin_feat, edges_list, edges_label, dimensions = 128, GCN=True)
                Acc.append(accuracy)
                Adj.append(adjust)
            average_acc = sum(Acc)/10
            average_adj = sum(Adj)/10
            print("----Epoch: %d -----Loss = %.4f----Accuracy = %.4f ----ADJ Score = %.4f"%(epoch, object_function, average_acc, average_adj))
            if pre_acc < average_acc:
                max_accuracy = average_acc
                max_adjust = average_adj
                pre_acc = average_acc
                torch.save(model.state_dict(),'./model/model.pt')
        print(" ----Max Accuracy : %.4f ---- MAx ADJ Score: %.4f ----"%(max_accuracy, max_adjust))

    def data_load(self):
        path = "./Sampling_graph/Datasets_With_Attributes/"+ os.path.basename(self.path) + ".graph"
        mul_nets, merge_nets, pos_edge_list, neg_edge_list, nodes_attr = Reader.data_load(path)
        mult_graphs = Graph2Coo.graphs2coo(mul_nets)
        graph_list = []
        for g in mult_graphs:
            x = torch.tensor(list(nodes_attr.values()), dtype=torch.float).to('cpu')
            g.x = x
            graph_list.append(g)
        data_merge = Graph2Coo.graphs2coo([merge_nets])[0]
        x = torch.tensor(list(nodes_attr.values()), dtype=torch.float)
        data_merge.x = x
        dataset = CreatMyDataset(graph_list, '../Benchmark/')
        edges_list, labels = get_selected_edges(pos_edge_list, neg_edge_list)
        return dataset, data_merge, edges_list, labels, nodes_attr

    def test_modal(self):
        path = './model/model.pt'
        dataset, data_merge, edges_list, edges_label, nodes_attr = self.data_load()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists(path):
            model = GATModel.Net(dataset, device)
            model.load_state_dict(torch.load(path))
            pre_x, normal_x, fin_feat = model.forward()
            for i in range(100):
                accuracy, adjust = eval.link_prediction(fin_feat, edges_list, edges_label)
                print(" ---- Accuracy : %.4f ---- ADJ Score: %.4f ----"%(accuracy, adjust))
        else:
            print('The model has saved in this file path!')

def get_selected_edges(pos_edge_list, neg_edge_list):
    edges = pos_edge_list + neg_edge_list
    labels = np.zeros(len(edges))
    labels[:len(pos_edge_list)] = 1
    return edges, labels

class CreatMyDataset(InMemoryDataset):
    def __init__(self, dataset, root, transform=None, pre_transform=None):
        self.data_list = dataset
        super(CreatMyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['CKM_Pyg.dataset']

    def download(self):
        pass

    def process(self):
        data_list = self.data_list
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
