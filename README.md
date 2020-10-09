# HMNE
Multiplex Network Embedding Model with High-Order Node Dependence
## Overview
Multiplex networks have been widely used in information diffusion, social network, transport and biology multi-omics, etc. It contains multiple types of relations between nodes, in which each type of relations is intuitively modelled as one layer. In real world, the formation of a type of relations may only depend on some attribute elements of nodes. Most existing multiplex network embedding methods only focus on intra-layer and inter-layer structural information, while neglecting this dependence between attributes and local topology of nodes. Attributes that are irrelevant to the network structure could affect the embedding quality of multiplex networks. To address this problem, we propose a novel multiplex network embedding model with high-order node dependence, called HMNE. HMNE simultaneously considers three node dependencies: 1) intra-layer high-order proximity of nodes, 2) inter-layer dependence in respect of nodes, 3) the dependence between node attributes and its local topology. In the intra-layer embedding phase, we present a symmetric graph convolution-deconvolution model to embed high-order proximity information as the intra-layer embedding of nodes in an unsupervised manner. In the inter-layer embedding phase, we estimate the local structural complementarity of nodes as an embedding constraint of inter-layers dependence. Through these two phases, we can achieve the disentanglement representation of node attributes, which can be treated as a fined-grained semantic dependence on its local topology. In node attributes restructure phase, we perform a linear fusion of attribute disentanglement representations for each node as a reconstruction of original attributes. Extensive experiments have been conducted on six real-world networks. The experimental results demonstrate that the proposed model outperforms the state-of-the-art methods in cross-domain link prediction and shared community detection tasks.

## Requirements
* Python 3.6
* Pytorch 1.5.0 + CPU
* Pytorch_geometric 1.5.0
* numpy 1.18.1
* pickle 0.7.4
* netwrokx 2.3
## Running The Code and Input Format

  ```Python Main.py```

## Parameters in Main.py are seted by Manual Maner

* ```run_flag = 'train'```

This code will be used to perform the **training** process.

* ```run_flag = 'test'```

This code will be used to perform the **testing** process.

* ```run_flag = 'compare'```

This code will be used to perform the **Baseline Methods**.
