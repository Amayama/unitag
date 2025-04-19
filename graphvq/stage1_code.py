import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization module for codebook construction as used in GraphVQ
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings  # Size of codebook (M^k)
        self.embedding_dim = embedding_dim    # Dimension of each code embedding
        self.commitment_cost = commitment_cost
        
        # Initialize the embedding table (codebook)
        # E_k is randomly initialized as mentioned in the figure
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, h_v_k):
        """
        h_v_k: Node representations from GNN at hop k
               Shape: [num_nodes, embedding_dim]
        
        Returns:
        - quantized: Quantized embeddings (closest codebook vectors)
        - codebook_indices: Indices of the closest codebook vectors
        - vq_loss: Vector quantization loss
        - commitment_loss: Commitment loss
        """
        # Flatten input if needed
        flat_input = h_v_k.view(-1, self.embedding_dim)
        
        # Calculate distances between input vectors and codebook vectors
        # For each node embedding, find the closest codebook vector
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook.weight**2, dim=1) - \
                    2 * torch.matmul(flat_input, self.codebook.weight.t())
        
        # Get the indices of the closest embeddings
        codebook_indices = torch.argmin(distances, dim=1)
        
        # Convert indices to one-hot encodings
        codebook_one_hot = F.one_hot(codebook_indices, num_classes=self.num_embeddings).float()
        
        # Quantize the input vectors (assign nodes to nearest code embedding)
        quantized = torch.matmul(codebook_one_hot, self.codebook.weight)
        quantized = quantized.view_as(h_v_k)
        
        # Compute the VQ loss
        vq_loss = F.mse_loss(quantized.detach(), h_v_k)
        
        # Compute the commitment loss
        commitment_loss = F.mse_loss(quantized, h_v_k.detach())
        
        # Straight-through estimator
        # Pass gradients from quantized embeddings to input embeddings
        quantized = h_v_k + (quantized - h_v_k).detach()
        
        total_loss = vq_loss + self.commitment_cost * commitment_loss
        
        return quantized, codebook_indices, total_loss


def construct_codebook(h_v_k, codebook_size, embedding_dim, commitment_cost=0.25):
    """
    Construct codebook and assign nodes to codes similar to GraphVQ
    
    Parameters:
    - h_v_k: Dictionary of node representations from GNN at hop k
             or node features X for hop 0
    - codebook_size: Size of the codebook (M^k)
    - embedding_dim: Dimension of embeddings
    - commitment_cost: Weight for commitment loss
    
    Returns:
    - Z_k: Codebook indices for each node (i.e., which code each node is assigned to)
    - E_k: The codebook embeddings
    - vq_module: The vector quantizer module for further training
    """
    # Convert node representations dictionary to tensor
    if isinstance(h_v_k, dict):
        nodes = list(h_v_k.keys())
        h_v_k_tensor = torch.stack([torch.tensor(h_v_k[v]) for v in nodes], dim=0)
    else:
        # If h_v_k is already a tensor (e.g., X for hop 0)
        h_v_k_tensor = torch.tensor(h_v_k)
    
    # Initialize the vector quantizer
    vq_module = VectorQuantizer(
        num_embeddings=codebook_size,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost
    )
    
    # Perform vector quantization
    quantized, codebook_indices, _ = vq_module(h_v_k_tensor)
    
    # Create Z_k: mapping from nodes to their assigned codebook indices
    Z_k = {}
    for i, node in enumerate(nodes) if isinstance(h_v_k, dict) else enumerate(range(len(h_v_k_tensor))):
        Z_k[node] = codebook_indices[i].item()
    
    # Get E_k: the codebook embeddings
    E_k = vq_module.codebook.weight.detach().numpy()
    
    return Z_k, E_k, vq_module


def hierarchical_codebook_construction(G, K, p_neighbors=10, initial_codebook_size=256, 
                                      gnn_model=None):
    """
    Hierarchical codebook construction for multiple hops as described in the figure
    
    Parameters:
    - G: Input graph with adjacency matrix A and node features X->NetworkX
    - K: Maximum number of hops
    - p_neighbors: Maximum number of neighbors to consider (p in the figure)
    - initial_codebook_size: Initial codebook size for hop 0
    - gnn_model: GNN model to use for node representation calculation
    
    Returns:
    - Codebooks and embeddings for all hops: {Z0, E0}, {Z1, E1}, ..., {ZK, EK}
    """
    # Extract node features X from graph G
    X = G.ndata['feat'] if hasattr(G, 'ndata') else np.array([G.nodes[v]['feat'] for v in G.nodes()])
    
    # Initialize results storage
    codebooks = []
    node_embeddings = {}
    
    # Hop 0: Initial codebook based on node features X
    Z0, E0, vq0 = construct_codebook(
        X, 
        codebook_size=initial_codebook_size,
        embedding_dim=X.shape[1]
    )
    codebooks.append((Z0, E0))
    node_embeddings[0] = {v: X[i] for i, v in enumerate(G.nodes())}
    
    # For each hop k from 1 to K
    for k in range(1, K+1):
        h_v_k = {}
        
        # For each node in the graph
        for v in G.nodes():
            # Get neighbors (limited to p)
            neighbors = list(G.neighbors(v))[:p_neighbors]
            
            # Get embeddings from previous hop
            prev_embeddings = [node_embeddings[k-1][u] for u in neighbors]
            
            if len(prev_embeddings) == 0:
                # Handle nodes with no neighbors
                h_v_k[v] = node_embeddings[k-1][v]  # Use previous embedding
                continue
                
            # Apply GNN to get k-hop representation
            if gnn_model is None:
                # Simple aggregation if no GNN provided
                neighbor_embeds = torch.tensor(prev_embeddings)
                h_v_k[v] = torch.mean(neighbor_embeds, dim=0).numpy()
            else:
                # Use provided GNN model
                h_v_k[v] = gnn_model(
                    node=v,
                    neighbors=neighbors,
                    prev_embeddings=node_embeddings[k-1]
                )
        
        # Construct codebook for hop k
        codebook_size = initial_codebook_size * (k + 1)  # Increase size with hop
        embedding_dim = len(next(iter(h_v_k.values())))
        
        Z_k, E_k, vq_k = construct_codebook(
            h_v_k,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim
        )
        
        codebooks.append((Z_k, E_k))
        
        # Update node embeddings for this hop
        node_embeddings[k] = {}
        for v in G.nodes():
            # Get the codebook embedding for this node
            node_embeddings[k][v] = E_k[Z_k[v]]
    
    return codebooks, node_embeddings