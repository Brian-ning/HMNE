import networkx as nx
import torch
import torch_geometric.data
from torch_geometric.data import InMemoryDataset

def graphs2coo(graphs, nodes_attr):
    gs = []
    for g in graphs:
        nx.set_node_attributes(g, nodes_attr, 'x') # 对 networkx创建的图的每个节点设置属性
        nx.set_edge_attributes(g, 1, 'weight')
        gs.append(from_networkx(g)) # 从networkx图创建为Pyg中的data图
    return gs # 返回一个pyg的图列表

def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(sorted(G.nodes(data=True))):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item, dtype=torch.float32)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data

class CreatMyDataset(InMemoryDataset):
    ''' 将 pyg图的列表转化为pyg的Dataset结构，返回一个包含多个图的Dataset数据集'''
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