import os
import torch
import torch.nn as nn
import Reader
import Graph2Coo
import GATModel
import numpy as np
import Evaluation as eval

class train_model:
    def __init__(self, path, sampling, dimension):
        self.path = path
        self.sampling = sampling
        self.dimension = dimension
        self.data = None

    def train_AMNE(self):
        # 加载数据集
        dataset, edges_list, edges_label = self.data_load()
        number_graphs = len(dataset)
        model = GATModel.Net(dataset, number_graphs)

        # 设置目标函数和优化方法
        criterion2 = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01, lr=0.001)

        pre_acc = 0
        for epoch in range(1, 6001):
            pre_feat, decoder_H, fuse_feat, encoder_H, layer_label = model()
            optimizer.zero_grad()
            object_function1 = 0
            object_function2 = 0
            for i in range(number_graphs):
                label = np.ones(number_graphs)
                label[i] = 0
                label4layer = np.repeat(np.array([label]), dataset[i].x.size()[0], axis =0)
                object_function1 = object_function1 + 800 * criterion2(layer_label[i], torch.from_numpy(label4layer).float())
                object_function2 = object_function2 + torch.norm(pre_feat[i]-decoder_H[i], p=2)
            object_function3 = torch.norm(fuse_feat - dataset[0].x, p=2)
            object_function = object_function1 + object_function2 + object_function3
            Acc = []
            Adj = []
            for i in range(10):
                accuracy, adjust = eval.link_prediction(encoder_H, edges_list, edges_label, dimensions = 128, GCN=True)
                Acc.append(accuracy)
                Adj.append(adjust)
            average_acc = sum(Acc)/10
            average_adj = sum(Adj)/10
            print("----Epoch: %d -----Loss = %.4f : %.4f/ %.4f/ %.4f----Accuracy = %.4f ----ADJ Score = %.4f"%(epoch, object_function, object_function1, object_function2, object_function3, average_acc, average_adj))
            if pre_acc < average_acc:
                max_accuracy = average_acc
                max_adjust = average_adj
                pre_acc = average_acc
                Max_acc = Acc
                Max_adj = Adj
                torch.save(model.state_dict(),'./model/model.pt')
            object_function.backward()
            optimizer.step()
        print(" ----Max Accuracy : %.4f ---- MAx ADJ Score: %.4f ----"%(max_accuracy, max_adjust))
        print([(Max_acc[i], Max_adj[i]) for i in range(len(Max_acc))])

    def data_load(self):
        path = "./Sampling_graph/Datasets_With_Attributes/"+ os.path.basename(self.path) + ".graph"
        mul_nets, pos_edge_list, neg_edge_list, nodes_attr = Reader.data_load(path)
        graph_list = Graph2Coo.graphs2coo(mul_nets, nodes_attr)
        dataset = Graph2Coo.CreatMyDataset(graph_list, '../Benchmark/')
        edges_list, labels = get_selected_edges(pos_edge_list, neg_edge_list)
        return dataset, edges_list, labels

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
