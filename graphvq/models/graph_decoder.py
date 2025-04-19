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

class GraphDecoder(nn.Module):
    """
    Graph decoder to reconstruct graph structure from codebook embeddings
    """
    def __init__(self, embedding_dim, hidden_dim, num_nodes):
        super(GraphDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.node_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.edge_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_embeddings):
        """
        Reconstruct graph structure from node embeddings
        
        Args:
            node_embeddings: Final node embeddings from the tokenizer
            
        Returns:
            adj_matrix: Reconstructed adjacency matrix
            node_features: Reconstructed node features
        """
        # Reconstruct node features
        node_features = self.node_predictor(node_embeddings)
        
        # Reconstruct adjacency matrix
        adj_matrix = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:  # Exclude self-loops if not needed
                    edge_features = torch.cat([node_embeddings[i], node_embeddings[j]])
                    adj_matrix[i, j] = self.edge_predictor(edge_features).squeeze()
                    
        return adj_matrix, node_features

