# M3NE
A Multi-level and Multi-phase Representation Learning for Multiplex Network

## Overview
We propose a novel Multi-level and Multi-phase Multiplex Network Embedding model, called M3NE. We divide the comprehensive embedding of M3NE into three phases, which are a local embedding phase and a global phase. In the preprocessing phase, we ﬁrst obtain the estimation of the structural complementarity information of each node as the multiplexity property of nodes in diﬀerent layers. In the local embedding phase, a two-level strategy is designed to fuse intra-layer and inter-layer information of multiplex networks. 1) At the intra-layer embedding level, inspired by self-supervised learning, we present a graph convolution-deconvolution model to learn an intra-layer representation of nodes in an unsupervised manner. 2) At inter-layer embedding level, we utilize a preprocessing step to estimate the complementary information similarities of a node for each layer network structure. This multi-level embedding model can eﬀectively capture inter-layer information of nodes and reveal the multiplexity property of nodes. In the global embedding phase, we present a global autoencoder structure embedding framework. This framework is to solve the situation in which nodes have the same attributes in diﬀerent layers. Moreover, it can also quantify the dependence of node’s structure formation on node attribute information in each layer of the network.

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
