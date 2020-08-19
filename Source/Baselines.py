import os
import copy
import Reader
import numpy as np
import pickle as pk
import networkx as nx
import Evaluation as eval
import openne.graph as opgraph
from sklearn.externals import joblib
from BaselineMethods.MNE import main as PMNE
from BaselineMethods.ohmnet import ohmnet
import BaselineMethods.Node2vec as Node2vec
from BaselineMethods.MELL.MELL.MELL import MELL_model
from BaselineMethods.MNE import MNE, Random_walk, Node2Vec_LayerSelect

Parameter = {
    "p":1,
    "q":2,
    "num_walks":15,
    "walk_length":10,
    "dimensions":128,
}

class run:
    def __init__(self, path, dimension):
        self.path = path
        self.dimension = dimension

    def load_data(self):
        train_edges = []
        path_pk = "baselines.pkl"
        if os.path.exists(path_pk):
            print("The pkl file has existed!")
            with open(path_pk, 'rb') as f:
                (nx_graph, merge_graph, pos_edge_list, neg_edge_list, nodes_attr) = pk.load(f)
        else:
            path = "Sampling_graph/Datasets_With_Attributes/"+ os.path.basename(self.path) + ".graph"
            nx_graph, merge_graph, pos_edge_list, neg_edge_list, nodes_attr = Reader.data_load(path)
            with open(path_pk, 'wb') as f:
                pk.dump((nx_graph, merge_graph, pos_edge_list, neg_edge_list, nodes_attr), f)
        # 对网络中的节点标签进行修改，需要进行排序
        test_edges, test_labels = get_selected_edges(pos_edge_list, neg_edge_list)
        nodes = sorted(list(merge_graph.nodes()))
        if nodes[0] > 0:
            train_edges.extend([[i, e[0] - 1, e[1] - 1, 1] for i in range(len(nx_graph)) for e in nx_graph[i].edges()])
            train_merge = nx.relabel_nodes(merge_graph, lambda x: int(x) - 1)
            train_nxgraph = [nx.relabel_nodes(g, lambda x: int(x) - 1) for g in nx_graph]
            test_edges = [[e[0]-1, e[1]-1] for e in test_edges]
            nodes = list(train_merge.nodes())
        else:
            train_edges.extend([[i, e[0], e[1], 1] for i in range(len(nx_graph)) for e in nx_graph[i].edges()])
            train_nxgraph = copy.deepcopy(nx_graph)
            train_merge = copy.deepcopy(merge_graph)

        # 有的节点编号并不是连续的，下面语句是为了使节点的编号连续
        if isinstance(test_edges[0], list):
            restru_test_edges = [(e[0], e[1]) for e in test_edges]
        else:
            restru_test_edges = [(nodes.index(e[0]), nodes.index(e[1])) for e in test_edges]
        str_graph = nx.relabel_nodes(train_merge, lambda x: str(x))

        # 下面操作的是opennet定义的网络，为了使用现有的单层网络算法做对比
        G = opgraph.Graph()
        DG = str_graph.to_directed()
        G.read_g(DG)
        nx_para_graph = []
        for g in train_nxgraph:
            str_graph = nx.relabel_nodes(g, lambda x: str(x))
            G = opgraph.Graph()
            DG = str_graph.to_directed()
            G.read_g(DG)
            nx_para_graph.append(G)

        return train_nxgraph, restru_test_edges, train_merge, test_edges, train_edges, test_labels

    def baselines_run(self):
        # 表示学习参数设置阶段
        p = Parameter["p"]
        q = Parameter["q"]
        num_walks = Parameter["num_walks"]
        walk_length = Parameter["walk_length"]
        dimensions = self.dimension
        train_nxgraph, restru_test_edges, train_merge, test_edges, train_edges, test_labels = self.load_data()
        # 1 Ohmnet method
        path = './compare/ohmnet.pkl'
        if os.path.exists(path):
            with open('./compare/ohmnet.pkl', 'rb') as f:
                ohmnet_model = pk.load(f)
        else:
            ohmnet_model = self.Ohmnet_methd(train_nxgraph, p, q, num_walks, walk_length, dimensions)
            with open('./compare/ohmnet.pkl', 'wb') as f:
                pk.dump(ohmnet_model, f)
        Acc = []
        Adj = []
        for i in range(100):
            accuracy, adjust = eval.link_prediction(ohmnet_model, test_edges, test_labels, dimensions = 128)
            Acc.append(accuracy)
            Adj.append(adjust)
        average_acc = sum(Acc)/100
        average_adj = sum(Adj)/100
        print("----Algorithm: %s -----Accuracy = %.4f ----ADJ Score = %.4f"%("Ohmnet", average_acc, average_adj))

        # 2 PMNE method
        path = './compare/pmne.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                [pmne_model_1, pmne_model_2, pmne_model_3] = pk.load(f)
        else:
            pmne_model_1, pmne_model_2, pmne_model_3 = self.PMNE(train_nxgraph, test_edges, restru_test_edges, p, q, num_walks, walk_length)
            with open(path, 'wb') as f:
                pk.dump([pmne_model_1,pmne_model_2,pmne_model_3], f)
        str_model_2 = {}
        for k,v in pmne_model_2.items():
            str_model_2[str(k)] = v
        pmne_model_2 = str_model_2
        Acc1 = []
        Adj1 = []
        Acc2 = []
        Adj2 = []
        Acc3 = []
        Adj3 = []
        for i in range(100):
            accuracy, adjust = eval.link_prediction(pmne_model_1.wv, test_edges, test_labels, dimensions = 128)
            Acc1.append(accuracy)
            Adj1.append(adjust)
            accuracy, adjust = eval.link_prediction(pmne_model_2, test_edges, test_labels, dimensions = 129)
            Acc2.append(accuracy)
            Adj2.append(adjust)
            accuracy, adjust = eval.link_prediction(pmne_model_3.wv, test_edges, test_labels, dimensions = 128)
            Acc3.append(accuracy)
            Adj3.append(adjust)
        average_acc1 = sum(Acc1)/100
        average_adj1 = sum(Adj1)/100
        average_acc2 = sum(Acc2)/100
        average_adj2 = sum(Adj2)/100
        average_acc3 = sum(Acc3)/100
        average_adj3 = sum(Adj3)/100
        print("----Algorithm: %s -----Accuracy = %.4f ----ADJ Score = %.4f"%("Pmne1", average_acc1, average_adj1))
        print("----Algorithm: %s -----Accuracy = %.4f ----ADJ Score = %.4f"%("Pmne2", average_acc2, average_adj2))
        print("----Algorithm: %s -----Accuracy = %.4f ----ADJ Score = %.4f"%("Pmne3", average_acc3, average_adj3))

        # 3 MNE method
        path = './compare/mne.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                mne_method = pk.load(f)
        else:
            mne_method = self.MNE_method(train_edges)
            with open(path, 'wb') as f:
                pk.dump(mne_method, f)
        Acc = []
        Adj = []
        for i in range(100):
            accuracy, adjust = eval.link_prediction(mne_method, test_edges, test_labels, dimensions = 128)
            Acc.append(accuracy)
            Adj.append(adjust)
        average_acc = sum(Acc)/100
        average_adj = sum(Adj)/100
        print("----Algorithm: %s -----Accuracy = %.4f ----ADJ Score = %.4f"%("Mne", average_acc, average_adj))

        # 4 MELL method
        path = './compare/mell.pkl'
        if os.path.exists(path):
            mell_method = joblib.load(path)
            with open(path, 'rb') as f:
                mell_method = pk.load(f)
        else:
            mell_method, N = self.MELL(train_nxgraph, train_merge, train_edges)
            VS = np.sum(mell_method.resVH, axis=0)
            VT = np.sum(mell_method.resVT, axis=0)
            v_embedding = np.add(VT, VS)
            fin_dict = {}
            for i in range(N):
                fin_dict[str(i)] = v_embedding[i]
            mell_method = fin_dict
            with open(path, 'wb') as f:
                    pk.dump(mell_method, f)
        Acc = []
        Adj = []
        for i in range(100):
            accuracy, adjust = eval.link_prediction(mell_method, test_edges, test_labels, dimensions = 128)
            Acc.append(accuracy)
            Adj.append(adjust)
        average_acc = sum(Acc)/100
        average_adj = sum(Adj)/100
        print("----Algorithm: %s -----Accuracy = %.4f ----ADJ Score = %.4f"%("Mell", average_acc, average_adj))


    def Ohmnet_methd(self, train_nxgraph, p, q, num_walks, walk_length, dimensions):
        # #2# Ohmnet 实现多层网络嵌入 Bioinformatics'2017
        ohmnet_walks = []
        orignal_walks = []
        LG = copy.deepcopy(train_nxgraph)
        on = ohmnet.OhmNet(LG, p=p, q=q, num_walks=num_walks,
            walk_length=walk_length, dimension=dimensions,
            window_size=10, n_workers=8, n_iter=5,out_dir= '.')
        for ns in on.embed_multilayer():
            orignal_walks.append(ns)
            on_walks = [n.split("_")[2] for n in ns]
            ohmnet_walks.append([str(step) for step in on_walks])
        Ohmnet_model = Node2vec.N2V.learn_embeddings(ohmnet_walks, dimensions, workers = 5, window_size=10, niter=5)
        return Ohmnet_model

    def PMNE(self, train_nxgraph, test_edges, restru_test_edges, p, q, num_walks, walk_length):
        # #4# PMNE的3种算法
        merged_networks = dict()
        merged_networks['training'] = dict()
        merged_networks['test_true'] = dict()
        merged_networks['test_false'] = dict()
        for index, g in enumerate(train_nxgraph):
            merged_networks['training'][index] = set(g.edges())
            merged_networks['test_true'][index] = restru_test_edges[index]
            merged_networks['test_false'][index] = test_edges[index][len(test_edges):]

        model_1, model_2, model_3 = Evaluate_PMNE_methods(merged_networks, False, p, q, num_walks, walk_length, self.dimension)
        return model_1, model_2, model_3

    def MNE_method(self, train_edges):
        edge_data_by_type = {}
        all_edges = list()
        all_nodes = list()
        for e in train_edges:
            if e[0] not in edge_data_by_type:
                edge_data_by_type[e[0]]=list()
            edge_data_by_type[e[0]].append((e[1],e[2]))
            all_edges.append((e[1], e[2]))
            all_nodes.append(e[1])
            all_nodes.append(e[2])
        all_nodes = list(set(all_nodes))
        all_edges = list(set(all_edges))
        edge_data_by_type['Base'] = all_edges
        MNE_model = MNE.train_model(edge_data_by_type)
        local_model = dict()
        for pos in range(len(MNE_model['index2word'])):
            local_model[MNE_model['index2word'][pos]] = MNE_model['base'][pos]
        return local_model

    def MELL(self, train_nxgraph, train_merge, train_edges):
        L = len(train_nxgraph)
        N = max([int(n) for n in train_merge.nodes()])+1
        N = max(N, train_merge.number_of_nodes()) # 为了构造邻接矩阵需要找到行的标准
        directed = True
        d = 128
        k = 3
        lamm = 10
        beta = 1
        gamma = 1
        MELL_wvecs = MELL_model(L, N, directed, train_edges, d, k, lamm, beta, gamma)
        MELL_wvecs.train(100) # 之前是500，但是有的数据集500会报错，因此设置为30
        return MELL_wvecs, N


def get_selected_edges(pos_edge_list, neg_edge_list):
    edges = pos_edge_list + neg_edge_list
    labels = np.zeros(len(edges))
    labels[:len(pos_edge_list)] = 1
    return edges, labels

def Evaluate_PMNE_methods(input_network, directed, p, q, num_walks, walk_length, dimensions):
    # we need to write codes to implement the co-analysis method of PMNE
    print('Start to analyze the PMNE method')
    training_network = input_network['training']
    test_network = input_network['test_true']
    false_network = input_network['test_false']
    all_network = list()
    all_test_network = list()
    all_false_network = list()
    all_nodes = list()
    for edge_type in training_network:
        for edge in training_network[edge_type]:
            all_network.append(edge)
            if edge[0] not in all_nodes:
                all_nodes.append(edge[0])
            if edge[1] not in all_nodes:
                all_nodes.append(edge[1])
        for edge in test_network[edge_type]:
            all_test_network.append(edge)
        for edge in false_network[edge_type]:
            all_false_network.append(edge)
    # print('We are working on method one')
    all_network = set(all_network)
    G = Random_walk.RWGraph(MNE.get_G_from_edges(all_network), directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    model_one = MNE.train_deepwalk_embedding(walks, size=dimensions)

    # print('We are working on method two')
    all_models = list()
    for edge_type in training_network:
        tmp_edges = training_network[edge_type]
        tmp_G = Random_walk.RWGraph(MNE.get_G_from_edges(tmp_edges), directed, p, q)
        tmp_G.preprocess_transition_probs()
        walks = tmp_G.simulate_walks(num_walks, walk_length)
        tmp_model = MNE.train_deepwalk_embedding(walks, size=int(dimensions/len(training_network)))
        all_models.append(tmp_model)
    model_two = merge_PMNE_models(all_models, all_nodes, 8)

    # print('We are working on method three')
    tmp_graphs = list()
    for edge_type in training_network:
        tmp_G = MNE.get_G_from_edges(training_network[edge_type])
        tmp_graphs.append(tmp_G)
    MK_G = Node2Vec_LayerSelect.Graph(tmp_graphs, p, q, 0.5)
    MK_G.preprocess_transition_probs()
    MK_walks = MK_G.simulate_walks(num_walks, walk_length)
    model_three = MNE.train_deepwalk_embedding(MK_walks)
    # method_three_performance = get_AUC(model_three, all_test_network, all_false_network)

    return model_one, model_two, model_three

def merge_PMNE_models(input_all_models, all_nodes, dimensions):
    all_nodes = [str(i) for i in all_nodes]
    final_model = dict()
    for tmp_model in input_all_models:
        for node in all_nodes:
            if node in final_model:
                if node in tmp_model.wv.index2word:
                    final_model[node] = np.concatenate((final_model[node], tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]), axis=0)
                else:
                    final_model[node] = np.concatenate((final_model[node], np.zeros([dimensions])), axis=0)
            else:
                if node in tmp_model.wv.index2word:
                    final_model[node] = tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]
                else:
                    final_model[node] = np.zeros([dimensions])
    return final_model


