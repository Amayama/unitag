import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
import os
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import to_dense_adj
import argparse



class GNNEncoder(nn.Module):
    """
    GNN encoder for k-hop node representation calculation
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, e_prev, edge_index, batch=None):
        """
        Compute node representations for the current hop
        
        Args:
            e_prev: Node embeddings from previous hop
            edge_index: Graph connectivity in COO format
            batch: Batch assignment for nodes
            
        Returns:
            Node representations for current hop
        """
        x = self.conv1(e_prev, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
    def sample_neighbors(self, edge_index, num_nodes, max_size=20):
        """
        Sample neighbors for each node with maximum size p
        
        Args:
            edge_index: Graph connectivity in COO format
            num_nodes: Number of nodes in the graph
            max_size: Maximum number of neighbors to sample
            
        Returns:
            Sampled edge index
        """
        sampled_edges = []
        for node in range(num_nodes):
            # Find neighbors of the current node
            neighbors = edge_index[1][edge_index[0] == node]
            
            # If neighbors exceed max_size, randomly sample max_size neighbors
            if len(neighbors) > max_size:
                perm = torch.randperm(len(neighbors))
                neighbors = neighbors[perm[:max_size]]
            
            # Add edges to sampled edges
            for neighbor in neighbors:
                sampled_edges.append([node, neighbor.item()])
        
        return torch.tensor(sampled_edges).t().contiguous()

