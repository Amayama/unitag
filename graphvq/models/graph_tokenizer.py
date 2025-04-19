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
from .GNNEncoder import *
from .GraphDecoder import *


class GraphTokenizer(nn.Module):
    """
    Hierarchical Graph Vector Quantizer (GraphVQ) for tokenizing graphs
    """
    def __init__(self, input_dim, hidden_dim, K, M_list, p, num_nodes):
        """
        Initialize the Graph Tokenizer
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            K: Maximum number of hops
            M_list: List of codebook sizes for each hop [M^0, M^1, ..., M^K]
            p: Maximum number of neighbors to sample
            num_nodes: Number of nodes in the graph
        """
        super(GraphTokenizer, self).__init__()
        self.K = K
        self.M_list = M_list
        self.p = p
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # For storing the last assignments for external access (wandb logging)
        self.last_assignments = []
        
        # Initialize GNN encoders for each hop
        self.gnn_encoders = nn.ModuleList([
            GNNEncoder(
                input_dim if k == 0 else hidden_dim, 
                hidden_dim, 
                hidden_dim
            ) for k in range(K)
        ])
        
        # Initialize codebooks for each hop
        self.register_buffer('codebooks', torch.zeros(K, max(M_list), hidden_dim))
        
        # Initialize decoder
        self.decoder = GraphDecoder(hidden_dim, hidden_dim, num_nodes)
        
    def build_codebook(self, embeddings, M, k):
        """
        Build or update codebook for the k-th hop
        
        Args:
            embeddings: Node embeddings for current hop
            M: Codebook size for current hop
            k: Current hop index
            
        Returns:
            Z_k: Codebook indices
            E_k: Codebook embeddings
        """
        # Use K-means clustering to create codebook
        from sklearn.cluster import KMeans
        
        # Convert embeddings to numpy for K-means
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=M, random_state=0).fit(embeddings_np)
        
        # Get cluster centers as codebook embeddings
        E_k = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(embeddings.device)
        
        # Get cluster assignments as codebook indices
        Z_k = torch.tensor(kmeans.labels_, dtype=torch.long).to(embeddings.device)
        
        # Update codebook for this hop
        self.codebooks[k, :M] = E_k
        
        return Z_k, E_k
    
    def assign_nearest(self, embeddings, codebook):
        """
        Assign each embedding to the nearest codebook entry
        
        Args:
            embeddings: Node embeddings
            codebook: Codebook embeddings
            
        Returns:
            Quantized embeddings
        """
        # Calculate distances between embeddings and codebook entries
        distances = torch.cdist(embeddings, codebook)
        
        # Find nearest codebook entry for each embedding
        indices = torch.argmin(distances, dim=1)
        
        # Assign nearest codebook embedding
        quantized = torch.index_select(codebook, 0, indices)
        
        return quantized, indices
    
    
    def calculate_loss(self, edge_index, X, E, assignments, lambda_vq=0.1, lambda_commit=0.1):
        """
        Calculate combined loss function
        
        Args:
            edge_index: Graph connectivity in COO format
            X: Node features
            E: List of embeddings for each hop
            assignments: List of tuples (h_k, quantized, indices) for each hop
            lambda_vq: Weight for VQ loss
            lambda_commit: Weight for commitment loss
            
        Returns:
            Combined loss value and dictionary of individual loss components
        """
        # Convert edge_index to dense adjacency matrix for reconstruction loss
        adj_matrix = to_dense_adj(edge_index, max_num_nodes=self.num_nodes).squeeze(0)
        
        # Decode graph from final embeddings
        recon_adj, recon_X = self.decoder(E[-1])
        
        # Reconstruction loss
        loss_recon_x = F.mse_loss(recon_X, X)
        loss_recon_adj = F.binary_cross_entropy(recon_adj, adj_matrix)
        loss_recon = loss_recon_x + loss_recon_adj
        
        # VQ loss and commitment loss
        loss_vq = 0
        loss_commit = 0
        
        for h_k, quantized, _ in assignments:
            # VQ loss: distance between encoder output and codebook embeddings
            loss_vq += F.mse_loss(h_k, quantized.detach())
            
            # Commitment loss: encourage encoder to commit to codebook entries
            loss_commit += F.mse_loss(h_k.detach(), quantized)
        
        # Combined loss
        total_loss = loss_recon + lambda_vq * loss_vq + lambda_commit * loss_commit
        
        # Create a dictionary of loss components for logging
        loss_components = {
            'total_loss': total_loss.item(),
            'recon_loss': loss_recon.item(),
            'recon_x_loss': loss_recon_x.item(),
            'recon_adj_loss': loss_recon_adj.item(),
            'vq_loss': loss_vq.item(),
            'commit_loss': loss_commit.item()
        }
        
        return total_loss, loss_components
    
    def encode_graph(self, edge_index, X):
        """
        Encode graph into tokens
        
        Args:
            edge_index: Graph connectivity in COO format
            X: Node features
            
        Returns:
            Z_list: List of token indices for each hop
        """
        with torch.no_grad():
            # Initialize: E^0 = X
            E = [X]
            Z_list = []
            
            for k in range(self.K):
                # Calculate k-hop node representations
                h_k = self.gnn_encoders[k](
                    e_prev=E[-1], 
                    edge_index=self.gnn_encoders[k].sample_neighbors(
                        edge_index=edge_index, 
                        num_nodes=self.num_nodes, 
                        max_size=self.p
                    )
                )
                
                # Assign to nearest codebook entry
                _, indices = self.assign_nearest(h_k, self.codebooks[k, :self.M_list[k]])
                Z_list.append(indices)
                
                # Get quantized embeddings for next hop
                quantized = torch.index_select(self.codebooks[k, :self.M_list[k]], 0, indices)
                E.append(quantized)
            
            return Z_list
    
    def decode_graph(self, Z_list):
        """
        Decode graph from tokens
        
        Args:
            Z_list: List of token indices for each hop
            
        Returns:
            adj_matrix: Reconstructed adjacency matrix
            node_features: Reconstructed node features
        """
        with torch.no_grad():
            # Get final hop embeddings
            final_embeddings = torch.zeros(self.num_nodes, self.hidden_dim).to(Z_list[0].device)
            
            for i in range(self.num_nodes):
                k = self.K - 1  # Use the final hop
                codebook_idx = Z_list[k][i].item()
                final_embeddings[i] = self.codebooks[k, codebook_idx]
            
            # Decode graph from embeddings
            adj_matrix, node_features = self.decoder(final_embeddings)
            
            return adj_matrix, node_features
