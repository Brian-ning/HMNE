import random

import networkx as nx
import numpy as np
import scipy.stats


def kl_divergence(p, q):
    return scipy.stats.entropy(p, q)


def js_divergence(p, q):
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


class RWGraph():
    def __init__(self, nx_G_all, node_type=None):
        self.G_all = nx_G_all
        self.node_type = node_type
        self.all_node = list(nx_G_all[-1].nodes())
        self.layer_transition_prob = dict()
        num_layers = self.num_layers = len(nx_G_all)

        for layer in range(num_layers - 1):
            nx_G_all[layer].add_nodes_from(self.all_node)

        for node in self.all_node:
            cur_transition_prob = np.zeros((num_layers, num_layers))

            for layer1 in range(num_layers):
                neighbors1 = list(nx_G_all[layer1][node].keys())
                cur_transition_prob[layer1][layer1] = 1
                if len(neighbors1) == 0:
                    continue
                for layer2 in range(layer1 + 1, num_layers):
                    neighbors2 = list(nx_G_all[layer2][node].keys())
                    if len(neighbors2) == 0:
                        continue
                    all_neighbors = neighbors1 + neighbors2
                    all_neighbors = list(set(all_neighbors))
                    p1 = np.zeros(len(all_neighbors))
                    p2 = np.zeros(len(all_neighbors))
                    for neighbor in neighbors1:
                        p1[all_neighbors.index(neighbor)] = 1
                    for neighbor in neighbors2:
                        p2[all_neighbors.index(neighbor)] = 1
                    cur_transition_prob[layer1][layer2] = (0.7 - js_divergence(p1, p2)) / 0.7
                    cur_transition_prob[layer2][layer1] = cur_transition_prob[layer1][layer2]

            cur_transition_prob = cur_transition_prob / cur_transition_prob.sum(axis=1).reshape((num_layers, 1))
            self.layer_transition_prob[node] = cur_transition_prob
        print("finish RWGraph init")

    def walk(self, walk_length, start, walk_layer, schema=None):
        # Simulate a random walk starting from start node in layer i.
        G_all = self.G_all

        rand = random.Random()

        if schema:
            schema_items = schema.split('-')
            assert schema_items[0] == schema_items[-1]

        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            neighbors = list(G_all[walk_layer][cur].keys())
            if schema == None:
                candidates = neighbors
            else:
                for node in neighbors:
                    if self.node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                        candidates.append(node)
            if candidates:
                walk.append(rand.choice(candidates))
            else:
                break
        return [str(node) for node in walk]

    def simulate_walks(self, num_walks, walk_length, walk_layer, schema=None):
        G_all = self.G_all
        walks = []
        nodes = self.all_node
        num_layers = self.num_layers
        # print('Walk iteration:')
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                if len(list(G_all[walk_layer][node].keys())) == 0:
                    continue
                if schema == None or schema.split('-')[0] == self.node_type[node]:
                    walks.append(self.walk(walk_length, node, walk_layer, schema=schema))
        for walk_iter in range(2*num_layers):
            random.shuffle(nodes)
            cur_transition_prob = self.layer_transition_prob[nodes[0]][walk_layer]
            for node in nodes:
                cur_transition_prob = (cur_transition_prob * 0.5 + self.layer_transition_prob[node][walk_layer] * 0.5)
                choose_layer = int(np.random.choice(num_layers, p=cur_transition_prob))
                if schema == None or schema.split('-')[0] == self.node_type[node]:
                    walks.append(self.walk(walk_length, node, choose_layer, schema=schema))
        return walks
