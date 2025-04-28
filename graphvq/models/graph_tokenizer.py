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
from models.graph_decoder import *
from models.graph_encoder import *
from models.graph_llm import *
from models.graph_tokenizer import *
from models.graph_tokenizer import *

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
        # Move embeddings to CPU for K-means
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Apply K-means clustering
        try:
            from sklearn.cluster import KMeans
            
            # Check if we have enough unique embeddings
            unique_count = len(np.unique(embeddings_np, axis=0))
            cluster_count = min(M, unique_count)
            
            if cluster_count < 2:
                # Not enough unique embeddings for clustering
                # Create a simple codebook with the mean embedding
                mean_embedding = embeddings.mean(dim=0, keepdim=True)
                E_k = mean_embedding.repeat(M, 1)
                Z_k = torch.zeros(embeddings.size(0), dtype=torch.long, device=embeddings.device)
            else:
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit(embeddings_np)
                
                # Get cluster centers as codebook embeddings (move back to device)
                E_k = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(embeddings.device)
                
                # Pad E_k to the desired size M if needed
                if E_k.size(0) < M:
                    padding = torch.randn(M - E_k.size(0), E_k.size(1), device=E_k.device) * 0.1
                    padding = padding + E_k.mean(dim=0, keepdim=True)
                    E_k = torch.cat([E_k, padding], dim=0)
                
                # Get cluster assignments as codebook indices
                Z_k = torch.tensor(kmeans.labels_, dtype=torch.long).to(embeddings.device)
            
        except Exception as e:
            print(f"Error in K-means clustering: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: Create a simple codebook with random entries
            E_k = torch.randn(M, embeddings.size(1), device=embeddings.device)
            
            # Assign random indices
            Z_k = torch.randint(0, M, (embeddings.size(0),), device=embeddings.device)
        
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
            quantized: Quantized embeddings
            indices: Indices of nearest codebook entries
        """
        # Calculate distances between embeddings and codebook entries
        distances = torch.cdist(embeddings, codebook)
        
        # Find nearest codebook entry for each embedding
        indices = torch.argmin(distances, dim=1)
        
        # Assign nearest codebook embedding
        quantized = torch.index_select(codebook, 0, indices)
        
        return quantized, indices
    
    def train_tokenizer(self, edge_index, X):
        """
        Train the graph tokenizer in a hierarchical manner
        
        Args:
            edge_index: Graph edge connections in COO format (2 x num_edges)
            X: Node feature matrix (num_nodes x input_dim)
            
        Returns:
            loss: Combined loss value
            E: List of embeddings for each hop
            Z: List of token indices for each hop
        """
        # Ensure consistent device usage
        device = X.device
        edge_index = edge_index.to(device)  # Move edge_index to same device as X
        
        # Initialize tracking variables
        E = [X]  # E^0 = X (node features are initial embeddings)
        Z = []   # Will store token indices for each hop
        assignments = []  # For calculating loss
        self.last_assignments = []  # Store for external access (e.g., wandb logging)
        
        # For each hop k
        for k in range(self.K):
            try:
                # Ensure GNN encoder is on the correct device
                self.gnn_encoders[k] = self.gnn_encoders[k].to(device)
                
                # Sample neighbors and ensure they're on the correct device
                sampled_edge_index = self.gnn_encoders[k].sample_neighbors(
                    edge_index=edge_index, 
                    num_nodes=min(X.size(0), self.num_nodes), 
                    max_size=self.p  # Control computational complexity by sampling
                ).to(device)
                
                # Step 1: Compute k-hop node representations
                # Use GNN to aggregate information from neighbors
                h_k = self.gnn_encoders[k](
                    e_prev=E[-1],  # Previous hop embeddings
                    edge_index=sampled_edge_index
                )
                
                # Step 2: Build codebook and get tokens
                # Use k-means clustering to create or update codebook for hop k
                Z_k, E_k = self.build_codebook(h_k, M=self.M_list[k], k=k)
                Z.append(Z_k)  # Store token indices
                
                # Ensure codebook embeddings are on correct device
                E_k = E_k.to(device)
                
                # Step 3: Assign nearest codebook embeddings
                # Vector quantization step - assign each node to closest codebook entry
                quantized, indices = self.assign_nearest(h_k, E_k)
                E.append(quantized)  # Store quantized embeddings
                
                # Store assignment information for loss calculation
                assignments.append((h_k, quantized, indices))
                self.last_assignments.append((h_k, quantized, indices))
                
            except Exception as e:
                print(f"Error processing hop {k}: {e}")
                import traceback
                traceback.print_exc()
                
                # Handle errors by adding None placeholders
                Z.append(None)
                E.append(E[-1] if E[-1] is not None else X)  # Use previous embedding or X as fallback
                assignments.append((None, None, None))
                self.last_assignments.append((None, None, None))
        
        # Step 4: Calculate combined loss
        try:
            loss, loss_components = self.calculate_loss(
                edge_index=edge_index, 
                X=X, 
                E=E, 
                assignments=assignments,
                lambda_vq=0.1,     # Weight for VQ loss
                lambda_commit=0.1  # Weight for commitment loss
            )
        except Exception as e:
            print(f"Error calculating loss: {e}")
            traceback.print_exc()
            # Return zero loss if calculation fails
            loss = torch.tensor(0.0, device=device)
            loss_components = {
                'total_loss': 0.0,
                'recon_loss': 0.0,
                'vq_loss': 0.0,
                'commit_loss': 0.0
            }
        
        return loss, E, Z
    def calculate_loss(self, edge_index, X, E, assignments, lambda_vq=0.1, lambda_commit=0.1):
        """
        Calculate the combined loss function
        
        Args:
            edge_index: Graph edge connections in COO format
            X: Node feature matrix
            E: List of embeddings for each hop
            assignments: List of (h_k, quantized, indices) tuples for each hop
            lambda_vq: Weight for VQ loss
            lambda_commit: Weight for commitment loss
            
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss components
        """
        # Check if we have valid assignments
        if len(assignments) == 0 or E[-1] is None:
            # Handle invalid assignments
            device = X.device
            zero_tensor = torch.tensor(0.0, device=device, requires_grad=True)
            loss_components = {
                'total_loss': zero_tensor.item(),
                'recon_loss': zero_tensor.item(),
                'recon_x_loss': zero_tensor.item(),
                'recon_adj_loss': zero_tensor.item(),
                'vq_loss': zero_tensor.item(),
                'commit_loss': zero_tensor.item()
            }
            return zero_tensor, loss_components
        
        try:
            # Get device from input tensors
            device = X.device
            
            # Convert edge_index to dense adjacency matrix for reconstruction loss
            # Ensure it's on the same device
            adj_matrix = to_dense_adj(edge_index.to(device), 
                                    max_num_nodes=self.num_nodes).squeeze(0).to(device)
            
            # Use the final hop embeddings to decode the graph
            recon_adj, recon_X = self.decoder(E[-1])
            
            # Ensure reconstructed tensors are on the right device
            recon_adj = recon_adj.to(device)
            recon_X = recon_X.to(device)
            
            # Handle feature dimension mismatch with projection layer
            if recon_X.size(1) != X.size(1):
                if not hasattr(self, 'feature_projection'):
                    # Create a projection layer on first use
                    self.feature_projection = nn.Linear(recon_X.size(1), X.size(1))
                    # Initialize with Xavier uniform 
                    nn.init.xavier_uniform_(self.feature_projection.weight)
                    nn.init.zeros_(self.feature_projection.bias)
                
                # Move projection layer to the correct device
                self.feature_projection = self.feature_projection.to(device)
                
                # Project reconstructed features to match original dimension
                recon_X_projected = self.feature_projection(recon_X)
                loss_recon_x = F.mse_loss(recon_X_projected, X)
            else:
                # If dimensions match, use original recon_X
                loss_recon_x = F.mse_loss(recon_X, X)
            
            # Ensure tensors for BCE loss have compatible shapes and are on same device
            if adj_matrix.shape != recon_adj.shape:
                # Use min size to handle shape mismatch
                min_size = min(adj_matrix.size(0), recon_adj.size(0))
                adj_matrix = adj_matrix[:min_size, :min_size]
                recon_adj = recon_adj[:min_size, :min_size]
            
            # Ensure adjacency matrices are on the correct device
            adj_matrix = adj_matrix.to(device)
            recon_adj = recon_adj.to(device)
            
            # Reconstruction loss for adjacency matrix
            loss_recon_adj = F.binary_cross_entropy(recon_adj, adj_matrix)
            loss_recon = loss_recon_x + loss_recon_adj
            
            # VQ loss and commitment loss
            loss_vq = torch.tensor(0.0, device=device, requires_grad=True)
            loss_commit = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Calculate VQ and commitment losses for each hop
            for h_k, quantized, _ in assignments:
                if h_k is not None and quantized is not None:
                    # Ensure tensors are on the correct device
                    h_k = h_k.to(device)
                    quantized = quantized.to(device)
                    
                    # VQ loss: distance between encoder output and codebook embeddings
                    # Use detach to stop gradients for the codebook
                    hop_vq_loss = F.mse_loss(h_k, quantized.detach())
                    loss_vq = loss_vq + hop_vq_loss
                    
                    # Commitment loss: encourages encoder output to stay close to codebook
                    # Use detach to stop gradients for the encoder
                    hop_commit_loss = F.mse_loss(h_k.detach(), quantized)
                    loss_commit = loss_commit + hop_commit_loss
            
            # Combined loss with safeguard against zero gradients
            total_loss = loss_recon
            
            # Only add VQ and commitment losses if they're non-zero
            if loss_vq.item() > 0:
                total_loss = total_loss + lambda_vq * loss_vq
            if loss_commit.item() > 0:
                total_loss = total_loss + lambda_commit * loss_commit
            
            # Check if total_loss requires grad
            if not total_loss.requires_grad:
                # Create a variable that requires grad
                dummy = torch.ones(1, device=device, requires_grad=True)
                total_loss = total_loss * dummy
            
            # Create loss components dictionary for logging
            loss_components = {
                'total_loss': total_loss.item(),
                'recon_loss': loss_recon.item(),
                'recon_x_loss': loss_recon_x.item(),
                'recon_adj_loss': loss_recon_adj.item(),
                'vq_loss': loss_vq.item(),
                'commit_loss': loss_commit.item()
            }
            
            return total_loss, loss_components
            
        except Exception as e:
            print(f"Error calculating loss: {e}")
            import traceback
            traceback.print_exc()
            
            # Return zero loss as fallback that requires grad
            device = X.device
            zero_tensor = torch.tensor(0.0, device=device, requires_grad=True)
            loss_components = {
                'total_loss': 0.0,
                'recon_loss': 0.0,
                'recon_x_loss': 0.0,
                'recon_adj_loss': 0.0,
                'vq_loss': 0.0,
                'commit_loss': 0.0
            }
            return zero_tensor, loss_components
    
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
                try:
                    # Sample neighbors
                    sampled_edge_index = self.gnn_encoders[k].sample_neighbors(
                        edge_index=edge_index, 
                        num_nodes=self.num_nodes, 
                        max_size=self.p
                    )
                    
                    # Calculate k-hop node representations
                    h_k = self.gnn_encoders[k](
                        e_prev=E[-1], 
                        edge_index=sampled_edge_index
                    )
                    
                    # Assign to nearest codebook entry
                    _, indices = self.assign_nearest(h_k, self.codebooks[k, :self.M_list[k]])
                    Z_list.append(indices)
                    
                    # Get quantized embeddings for next hop
                    quantized = torch.index_select(self.codebooks[k, :self.M_list[k]], 0, indices)
                    E.append(quantized)
                    
                except Exception as e:
                    print(f"Error encoding hop {k}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Add empty tensor as placeholder
                    Z_list.append(None)
            
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
            # Find the first non-None Z in Z_list
            valid_Z = None
            valid_k = -1
            for k in range(self.K):
                if k < len(Z_list) and Z_list[k] is not None and Z_list[k].numel() > 0:
                    valid_Z = Z_list[k]
                    valid_k = k
                    break
            
            if valid_Z is None:
                print("Warning: No valid Z in Z_list, returning zeros")
                return torch.zeros((self.num_nodes, self.num_nodes), device=self.codebooks.device), \
                       torch.zeros((self.num_nodes, self.hidden_dim), device=self.codebooks.device)
            
            # Get number of nodes from first valid Z
            num_nodes_in_Z = valid_Z.size(0)
            
            # Get final hop embeddings
            actual_num_nodes = min(num_nodes_in_Z, self.num_nodes)
            final_embeddings = torch.zeros(actual_num_nodes, self.hidden_dim, device=self.codebooks.device)
            
            # Use the last available hop for each node
            last_valid_k = valid_k
            for k in range(self.K):
                if k < len(Z_list) and Z_list[k] is not None and Z_list[k].numel() > 0:
                    last_valid_k = k
            
            # Get embeddings from the last valid hop
            if Z_list[last_valid_k] is not None and Z_list[last_valid_k].numel() > 0:
                z = Z_list[last_valid_k]
                actual_nodes = min(z.size(0), actual_num_nodes)
                
                for i in range(actual_nodes):
                    codebook_idx = z[i].item()
                    if codebook_idx < self.M_list[last_valid_k]:
                        final_embeddings[i] = self.codebooks[last_valid_k, codebook_idx]
            
            # Decode graph from embeddings
            adj_matrix, node_features = self.decoder(final_embeddings)
            
            return adj_matrix, node_features
