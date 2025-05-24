import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNEncoder(nn.Module):
    """
    GNN encoder for k-hop node representation calculation
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Main GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
        # Batch normalization for better training stability
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(output_dim)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # **FIX: Add input projection for dimension mismatch handling**
        self.input_projection = None
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, input_dim)  # Identity-like projection
        
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
        # **FIX: Validate input dimensions**
        if e_prev.size(1) != self.input_dim:
            print(f"Warning: Input dimension mismatch. Expected {self.input_dim}, got {e_prev.size(1)}")
            # If there's a significant mismatch, we need to handle it
            if e_prev.size(1) > self.input_dim:
                # Truncate if input is larger
                e_prev = e_prev[:, :self.input_dim]
                print(f"Truncated input to {self.input_dim} dimensions")
            else:
                # Pad if input is smaller
                padding_size = self.input_dim - e_prev.size(1)
                padding = torch.zeros(e_prev.size(0), padding_size, device=e_prev.device)
                e_prev = torch.cat([e_prev, padding], dim=1)
                print(f"Padded input to {self.input_dim} dimensions")
        
        # Apply optional input projection
        if self.input_projection is not None:
            e_prev = self.input_projection(e_prev)
        
        # First GCN layer
        x = self.conv1(e_prev, edge_index)
        
        # Apply batch normalization if we have enough samples
        if x.size(0) > 1:
            x = self.batch_norm1(x)
        
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        # Apply batch normalization if we have enough samples
        if x.size(0) > 1:
            x = self.batch_norm2(x)
        
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
        # **FIX: Better error handling and edge case management**
        if edge_index.numel() == 0:
            # Return empty edge index if no edges
            return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
        
        sampled_edges = []
        
        for node in range(num_nodes):
            # Find neighbors of the current node
            neighbors = edge_index[1][edge_index[0] == node]
            
            if len(neighbors) == 0:
                continue  # Skip nodes with no neighbors
            
            # If neighbors exceed max_size, randomly sample max_size neighbors
            if len(neighbors) > max_size:
                perm = torch.randperm(len(neighbors))
                neighbors = neighbors[perm[:max_size]]
            
            # Add edges to sampled edges
            for neighbor in neighbors:
                sampled_edges.append([node, neighbor.item()])
        
        if len(sampled_edges) == 0:
            # If no edges, return empty tensor with correct shape
            return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
        
        return torch.tensor(sampled_edges, device=edge_index.device).t().contiguous()
