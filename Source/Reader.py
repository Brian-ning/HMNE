#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, pickle, sys
import networkx as nx
import scipy.io as scio
import numpy as np
import pickle

### Assuming the input files are all pickle encoded networkx graph object ###

def data_load(path):
    if os.path.exists(path):
        # 加载已分好的缓存数据
        print("Loading multiplex networks from %s" % path)
        with open(path, 'rb') as f:
            cache_data = pickle.load(f)
            LG = cache_data['g_train']
            _pos_edge_list = cache_data['remove_list']
            _neg_edge_list = cache_data['ne_list']
        nodes_list = [int(v) for g in LG for v in g.nodes()]
        att_path = "Sampling_graph/Datasets_With_Attributes/Node_Attributes/"+ os.path.basename(path).split('.')[0] + "_nodes.txt"
        if os.path.exists(att_path):
            node_attr = load_attribbute(att_path, attribute = True, node_number = None)
        else:
            node_attr = load_attribbute('', attribute=False, node_number=max(nodes_list))
        flag = min(node_attr.keys())
        LG = node_label_to_int(LG, flag)
        multi_digraphs = [nx.DiGraph(g) for g in LG] # 出现图冻结的情况
        attr_dict = {}
        if min(node_attr.keys()) == 1:
            for key, value in node_attr.items():
                attr_dict[key-1] = list(value)
                for i in range(len(multi_digraphs)):
                    if key-1 not in multi_digraphs[i].nodes():
                        multi_digraphs[i].add_edge(key-1, key-1, weight=1)
        else:
            for key, value in node_attr.items():
                attr_dict[key] = list(value)
                for i in range(len(multi_digraphs)):
                    if key not in multi_digraphs[i].nodes():
                        multi_digraphs[i].add_edge(key, key, weight=1)

        if flag == 1:
            _pos_edge_list = [(int(e[0])-1, int(e[1])-1) for e in _pos_edge_list if int(e[0]) in nodes_list and int(e[1]) in nodes_list]
            _neg_edge_list = [(int(e[0])-1, int(e[1])-1) for e in _neg_edge_list if int(e[0]) in nodes_list and int(e[1]) in nodes_list]
        else:
            _pos_edge_list = [(int(e[0]), int(e[1])) for e in _pos_edge_list if int(e[0]) in nodes_list and int(e[1]) in nodes_list]
            _neg_edge_list = [(int(e[0]), int(e[1])) for e in _neg_edge_list if int(e[0]) in nodes_list and int(e[1]) in nodes_list]

        # merge_graph = merge_g(multi_digraphs)
        return multi_digraphs, _pos_edge_list, _neg_edge_list, attr_dict


def GKdata_load(LG, _pos_edge_list, _neg_edge_list):
    # 加载已分好的缓存数据
    print("Loading multiplex networks from %s")
    node_attr = GKload_attribbute('attr.txt', LG)
    flag = min(node_attr.keys())
    LG = node_label_to_int(LG, flag)
    multi_digraphs = [nx.DiGraph(g) for g in LG] # 出现图冻结的情况
    attr_dict = {}
    if min(node_attr.keys()) == 1:
        for key, value in node_attr.items():
            attr_dict[key-1] = list(value)
            for i in range(len(multi_digraphs)):
                if key-1 not in multi_digraphs[i].nodes():
                    multi_digraphs[i].add_edge(int(key-1), int(key-1), weight=1)
    else:
        for key, value in node_attr.items():
            attr_dict[key] = list(value)
            for i in range(len(multi_digraphs)):
                if key not in multi_digraphs[i].nodes():
                    multi_digraphs[i].add_edge(int(key), int(key), weight=1)

    if flag == 1:
        _pos_edge_list = [(int(e[0])-1, int(e[1])-1) for e in _pos_edge_list]
        _neg_edge_list = [(int(e[0])-1, int(e[1])-1) for e in _neg_edge_list]
    else:
        _pos_edge_list = [(int(e[0]), int(e[1])) for e in _pos_edge_list]
        _neg_edge_list = [(int(e[0]), int(e[1])) for e in _neg_edge_list]

    merge_graph = merge_g(multi_digraphs)
    return multi_digraphs, merge_graph, _pos_edge_list, _neg_edge_list, attr_dict

def node_label_to_int(graphs, flag):
    GS = []
    for g in graphs:
        if flag == 1:
            GS.append(nx.to_directed(nx.relabel_nodes(g, lambda x:int(x)-1)))
        else:
            GS.append(nx.to_directed(nx.relabel_nodes(g, lambda x:int(x))))
    return GS

def load_attribbute(att_path, attribute = True, node_number = None):
    nodes_attr = {}
    if attribute ==True:
        with open(att_path, 'rb') as f:
            nodes_attr_matrix = np.loadtxt(f, delimiter=' ', encoding='utf-8')
        for i in range(len(nodes_attr_matrix)):
            nodes_attr[nodes_attr_matrix[i][0]] = nodes_attr_matrix[i][1:]
    else:
        nodes_attr_matrix = np.random.rand(node_number, 16)
        for i in range(len(nodes_attr_matrix)):
            nodes_attr[i+1] = nodes_attr_matrix[i,:]
    return nodes_attr

def GKload_attribbute(att_path, LG):
    nodes_attr = {}
    max_nodes_set= set([v for g in LG for v in g.nodes()])
    if os.path.exists(att_path):
        with open(att_path, 'rb') as f:
            nodes_attr_matrix = np.loadtxt(f, delimiter=' ', encoding='utf-8')
        for i in range(len(nodes_attr_matrix)):
            nodes_attr[nodes_attr_matrix[i][0]] = nodes_attr_matrix[i][1:]
    else:
        nodes_attr_matrix = np.random.rand(len(max_nodes_set), 13)
        for i in range(len(nodes_attr_matrix)):
            nodes_attr[max_nodes_set[i]] = nodes_attr_matrix[i][:]

    return nodes_attr

def single_readG(path):
    if os.path.isfile(path) and path.endswith(".pickle"):
        g_need = pickle.load(open(path, "rb"))
        #g_need = max(nx.connected_component_subgraphs(g), key=len)
        return g_need
    else:
        sys.exit("##cannot find the pickle file from give path: " + path + "##")


def multi_readG(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        nx_graphs = []
        total_edges = 0
        for name in files:
            if name.endswith(".pickle"):
                ## Serialize to save the object.The Unserialization
                g_need = pickle.load(open(name, "rb"))
                #g_need = max(nx.connected_component_subgraphs(g), key=len)
                nx_graphs.append(g_need)
                total_edges += len(g_need.edges())
        return nx_graphs, total_edges
    else:
        sys.exit("##input path is not a directory##")


def multi_readG_with_Merg(path):
    if os.path.isdir(path):  # Judge whether this path is folder
        files = os.listdir(path)  # Get the file name list under this folder
        nx_graphs = []  # inistall the variable
        m_graph = -1
        total_edges = 0  # The total number of edges
        for name in files:
            if name.endswith("pickle"):  # Checking the file name
                if "merged_graph" in name:
                    m_graph = single_readG(path + '/' + name)
                else:
                    g_need = pickle.load(open(path + '/' + name, "rb"))
                    nx_graphs.append(g_need)
                    total_edges += len(g_need.edges())
        return m_graph, nx_graphs, total_edges

def weight(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        weight_dict = {}
        for name in files:
            if name.endswith('_info.txt'):
                for line in open(path+name):
                    (lay_a, lay_b, coef) = line.split(' ')
                    weight_dict[(lay_a, lay_b)] = float(coef)
    return weight_dict

def true_cluster(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        weight_dict = {}
        for name in files:
            if name.endswith('_true.mat'):
                data = scio.loadmat(path+name)

    return data['s_LNG']


def read_airline(path):
    if os.path.isdir(path):
        print("reading from " + path + "......")
        files  = os.listdir(path)
        nx_graphs = []
        airport_dst = {}
        airport_mapping = {}
        for name in files:
            if name.endswith('_networks.pickle'):
                print("found file " + name + "...")
                graphs = pickle.load(open(path + name, 'rb'))
                for key in graphs:
                    nx_graphs.append(graphs[key])

            elif name.endswith('_Features.pickle'):
                print("found file " + name + "...")
                airport_dst = pickle.load(open(path + name, 'rb'))
            elif name.endswith('_List_mapping.pickle'):
                print("found file " + name + "...")
                airport_mapping = pickle.load(open(path + name, 'rb'))

        #print(len(nx_graphs))
        return nx_graphs, airport_mapping, airport_dst

    else:
        sys.exit('Input path is not a directory')


def merge_g(nx_graphs):
    m_g = nx.Graph()
    for g in nx_graphs:
        m_g.add_nodes_from(g.nodes())
        m_g.add_edges_from(g.edges())

    return m_g

def read_f(filename):
    if os.path.isfile(filename) and filename.endswith(".edges"):
        print("reading from " + filename + "......")
        graph_dict = {}
        for line in open(filename):
            (src, layer_id, dst) = line.split()
            if layer_id not in graph_dict.keys():
                graph_dict[layer_id] = nx.Graph(name=layer_id)
                graph_dict[layer_id].add_edge(src, dst)
            else:
                graph_dict[layer_id].add_edge(src, dst)

        for i in graph_dict:
            f = open(i+'.txt', 'w+')
            for edge in graph_dict[i].edges():
                f.write(edge[0] + ' ' + edge[1] + '\n')
        return graph_dict

def read_test_dataset(filename):
    if os.path.isfile(filename) and filename.endswith("test.txt"):
        print("reading from " + filename + "......")
        pos_link = []
        for line in open(filename):
            (src, layer_id, dst) = line.split()
            pos_link.append((src, dst))

        return pos_link


if __name__ == '__main__':
    cached_fn = "baselines.pkl"
    graph_dict = read_f("./train.edges")
    nx_graphs = list(graph_dict.values())
    merge_graph = merge_g(nx_graphs)
    pos_link = read_test_dataset("./test.txt")
    g_nodes = [list(g.nodes()) for g in nx_graphs]
    neg_link = [e for e in nx.non_edges(merge_graph)]
    len_test = min([len(pos_link), len(neg_link)])
    pos_link = pos_link[:len_test]
    neg_link = neg_link[:len_test]
    nx_graph, merge_graph, pos_edge_list, neg_edge_list, nodes_attr = GKdata_load(nx_graphs, pos_link, neg_link)

    with open(cached_fn, 'wb') as f:
        pk.dump((nx_graph, merge_graph, pos_edge_list, neg_edge_list, nodes_attr), f)

