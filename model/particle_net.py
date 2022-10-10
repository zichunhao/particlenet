"""
Taken from https://github.com/rkansal47/mnist_graph_gan/blob/139a82282243a2b6cf201e9ee999a0a9a03e7b32/jets/particlenet.py
"""

import logging

import torch
from torch import nn

from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_cluster import knn_graph

import numpy as np
import torch.nn.functional as F


class ParticleNetEdgeNet(nn.Module):
    def __init__(self, in_size: int, layer_size: int):
        super(ParticleNetEdgeNet, self).__init__()

        layers = []

        layers.append(nn.Linear(in_size * 2, layer_size))
        layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for i in range(2):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __repr__(self) -> str:
        return '{}(nn={})'.format(self.__class__.__name__, self.model)


class ParticleNet(nn.Module):
    def __init__(
        self,
        num_hits: int,
        node_feat_size: int,
        num_classes: int = 1,  # a single score for signal vs background
        device: torch.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu'),
    ):
        super(ParticleNet, self).__init__()
        self.num_hits = num_hits
        self.node_feat_size = node_feat_size
        self.num_classes = num_classes
        self.device = device

        self.k = 16
        self.num_edge_convs = 3
        self.kernel_sizes = [64, 128, 256]
        self.fc_size = 256
        self.dropout = 0.1

        self.edge_nets = nn.ModuleList()
        self.edge_convs = nn.ModuleList()

        self.kernel_sizes.insert(0, self.node_feat_size)
        self.output_sizes = np.cumsum(self.kernel_sizes)

        self.edge_nets.append(ParticleNetEdgeNet(
            self.node_feat_size, self.kernel_sizes[1]))
        self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr='mean'))

        for i in range(1, self.num_edge_convs):
            # adding kernel sizes because of skip connections
            self.edge_nets.append(ParticleNetEdgeNet(
                self.output_sizes[i], self.kernel_sizes[i + 1]))
            self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr='mean'))

        self.fc1 = nn.Sequential(
            nn.Linear(self.output_sizes[-1], self.fc_size))

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.fc2 = nn.Linear(self.fc_size, self.num_classes)

        # logging.info("edge nets: ")
        # logging.info(self.edge_nets)

        logging.info("edge_convs: ")
        logging.info(self.edge_convs)

        logging.info("fc1: ")
        logging.info(self.fc1)

        logging.info("fc2: ")
        logging.info(self.fc2)

    def forward(
        self,
        x: torch.Tensor,
        ret_activations: bool = False,
        relu_activations: bool = False
    ) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.reshape(batch_size * self.num_hits, self.node_feat_size)
        zeros = torch.zeros(batch_size * self.num_hits,
                            dtype=int).to(self.device)
        zeros[torch.arange(batch_size) * self.num_hits] = 1
        batch = torch.cumsum(zeros, 0) - 1

        for i in range(self.num_edge_convs):
            # using only angular coords for knn in first edgeconv block
            edge_index = knn_graph(
                x[:, :2], self.k, batch) if i == 0 else knn_graph(x, self.k, batch)
            # concatenating with original features i.e. skip connection
            x = torch.cat((self.edge_convs[i](x, edge_index), x), dim=1)

        x = global_mean_pool(x, batch)
        x = self.fc1(x)

        if ret_activations:
            if relu_activations:
                return F.relu(x)
            else:
                return x    # for Frechet ParticleNet Distance
        else:
            x = self.dropout_layer(F.relu(x))
        
        if self.num_classes == 1:
            # binary classification
            return torch.sigmoid(self.fc2(x))
        else:
            # no softmax because pytorch cross entropy loss includes softmax
            return self.fc2(x)
