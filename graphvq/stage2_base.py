import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
import os
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import to_dense_adj
from models.graph_decoder import *
from models.graph_encoder import *
from models.graph_llm import *
from models.graph_tokenizer import *
from models.graph_tokenizer import *


# Example usage for training

def train_graph_tokenizer(tokenizer, data_loader, optimizer, device, epochs=100, log_wandb=True):
    """
    Train the graph tokenizer with detailed wandb logging
    
    Args:
        tokenizer: Graph tokenizer model
        data_loader: Data loader for graph datasets
        optimizer: Optimizer
        device: Device to run on
        epochs: Number of epochs
        log_wandb: Whether to log metrics to Weights & Biases
    """
    tokenizer.to(device)
    tokenizer.train()
    
    # Create a learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        total_loss = 0
        recon_loss_sum = 0
        vq_loss_sum = 0
        commit_loss_sum = 0
        recon_x_loss_sum = 0
        recon_adj_loss_sum = 0
        
        # Set tokenizer to training mode
        tokenizer.train()
        
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            batch = batch.to(device)
            edge_index, x = batch.edge_index, batch.x
            
            optimizer.zero_grad()
            
            # Train tokenizer and get loss components
            loss, E, Z = tokenizer.train_tokenizer(edge_index, x)
            
            # For better tracking, let's calculate loss components
            with torch.no_grad():
                _, loss_components = tokenizer.calculate_loss(
                    edge_index=edge_index, 
                    X=x, 
                    E=E, 
                    assignments=[(h_k, quantized, _) for h_k, quantized, _ in tokenizer.last_assignments]
                )
                
                total_loss += loss_components['total_loss']
                recon_loss_sum += loss_components['recon_loss']
                vq_loss_sum += loss_components['vq_loss']
                commit_loss_sum += loss_components['commit_loss']
                recon_x_loss_sum += loss_components['recon_x_loss']
                recon_adj_loss_sum += loss_components['recon_adj_loss']
            
            loss.backward()
            optimizer.step()
            
            # Log batch metrics to wandb
            if log_wandb and batch_idx % 10 == 0:  # Log every 10 batches to avoid cluttering
                wandb.log({
                    "batch": epoch * len(data_loader) + batch_idx,
                    "batch/total_loss": loss_components['total_loss'],
                    "batch/recon_loss": loss_components['recon_loss'],
                    "batch/recon_x_loss": loss_components['recon_x_loss'],
                    "batch/recon_adj_loss": loss_components['recon_adj_loss'],
                    "batch/vq_loss": loss_components['vq_loss'],
                    "batch/commit_loss": loss_components['commit_loss'],
                })
                
                # Log codebook utilization (histogram of token assignments)
                for k in range(tokenizer.K):
                    if len(Z) > k:
                        wandb.log({
                            f"codebook/hop_{k}_utilization": wandb.Histogram(Z[k].cpu().numpy())
                        })
                
                # Visualize codebook embeddings with PCA or t-SNE
                if batch_idx == 0 and epoch % 5 == 0:
                    try:
                        from sklearn.decomposition import PCA
                        
                        for k in range(tokenizer.K):
                            codebook = tokenizer.codebooks[k, :tokenizer.M_list[k]].cpu().numpy()
                            
                            if codebook.shape[0] > 2:  # Need at least 3 points for PCA
                                pca = PCA(n_components=2)
                                codebook_2d = pca.fit_transform(codebook)
                                
                                plt.figure(figsize=(8, 8))
                                plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], alpha=0.7)
                                plt.title(f"Hop {k} Codebook Embeddings (PCA)")
                                
                                wandb.log({f"codebook/hop_{k}_pca": wandb.Image(plt)})
                                plt.close()
                    except Exception as e:
                        print(f"Error generating PCA visualization: {e}")
        
        # Calculate average losses for the epoch
        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = recon_loss_sum / num_batches
        avg_vq_loss = vq_loss_sum / num_batches
        avg_commit_loss = commit_loss_sum / num_batches
        avg_recon_x_loss = recon_x_loss_sum / num_batches
        avg_recon_adj_loss = recon_adj_loss_sum / num_batches
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, VQ: {avg_vq_loss:.4f}, Commit: {avg_commit_loss:.4f}")
        
        # Log epoch metrics to wandb
        if log_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/recon_loss": avg_recon_loss,
                "train/vq_loss": avg_vq_loss,
                "train/commit_loss": avg_commit_loss,
                "train/recon_x_loss": avg_recon_x_loss,
                "train/recon_adj_loss": avg_recon_adj_loss,
                "train/learning_rate": optimizer.param_groups[0]['lr']
            })
            
            # Log model parameters as histograms (every 5 epochs to avoid overhead)
            if epoch % 5 == 0:
                for name, param in tokenizer.named_parameters():
                    if param.requires_grad:
                        wandb.log({f"parameters/{name}": wandb.Histogram(param.data.cpu().numpy())})
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = tokenizer.state_dict().copy()
            
            # Save best model checkpoint
            torch.save(best_model_state, "best_tokenizer_checkpoint.pt")
            if log_wandb:
                wandb.save("best_tokenizer_checkpoint.pt")
                wandb.log({"best_epoch": epoch, "best_loss": best_loss})
        
        # Every 10 epochs, save a checkpoint
        if epoch % 10 == 9 or epoch == epochs - 1:
            checkpoint_path = f"tokenizer_checkpoint_epoch_{epoch+1}.pt"
            torch.save(tokenizer.state_dict(), checkpoint_path)
            if log_wandb:
                wandb.save(checkpoint_path)
    
    # Restore best model
    if best_model_state is not None:
        tokenizer.load_state_dict(best_model_state)
    
    # Final model save
    torch.save(tokenizer.state_dict(), "final_tokenizer_checkpoint.pt")
    if log_wandb:
        wandb.save("final_tokenizer_checkpoint.pt")
    
    return tokenizer

def train_graph_llm(graph_llm, data_loader, optimizer, device, epochs=50, log_wandb=True, text_tokenizer=None):
    """
    Train the graph LLM with detailed wandb logging
    
    Args:
        graph_llm: Graph LLM model
        data_loader: Data loader for graph-text pairs
        optimizer: Optimizer
        device: Device to run on
        epochs: Number of epochs
        log_wandb: Whether to log metrics to Weights & Biases
        text_tokenizer: Text tokenizer for generating samples
    """
    graph_llm.to(device)
    graph_llm.train()
    
    # Create a learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        total_loss = 0
        token_accuracy = 0
        total_tokens = 0
        
        # Set model to training mode
        graph_llm.train()
        
        for batch_idx, batch in enumerate(data_loader):
            # Unpack batch: text tokens and graph
            prompt_tokens, graph = batch
            
            # Move data to device
            prompt_tokens = prompt_tokens.to(device)
            graph = graph.to(device)
            
            optimizer.zero_grad()
            
            # Get graph tokens from the tokenizer
            try:
                edge_index, x = graph.edge_index, graph.x
                Z_list = graph_llm.tokenizer.encode_graph(edge_index, x)
                
                # Convert to text tokens
                graph_tokens = graph_llm.graph_to_tokens(Z_list)
                
                # Combine with prompt tokens for input sequence
                # Reshape prompt_tokens if needed to match graph_tokens
                if len(prompt_tokens.shape) == 1:
                    prompt_tokens = prompt_tokens.unsqueeze(0)
                if len(graph_tokens.shape) == 1:
                    graph_tokens = graph_tokens.unsqueeze(0)
                
                # Make sure they have the same batch dimension
                batch_size = min(prompt_tokens.size(0), graph_tokens.size(0))
                prompt_tokens = prompt_tokens[:batch_size]
                graph_tokens = graph_tokens[:batch_size]
                
                # Combine tokens
                input_tokens = torch.cat([prompt_tokens, graph_tokens], dim=1)
                
                # Forward pass through the LLM
                outputs = graph_llm(input_tokens)
                
                # Calculate loss (next token prediction)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = input_tokens[..., 1:].contiguous()
                
                # Compute cross entropy loss
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Calculate token prediction accuracy
                with torch.no_grad():
                    preds = torch.argmax(shift_logits, dim=-1)
                    correct = (preds == shift_labels).float().sum()
                    token_accuracy += correct.item()
                    total_tokens += shift_labels.numel()
                
                # Backward pass and optimization
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(graph_llm.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                # Log batch metrics to wandb
                if log_wandb and batch_idx % 10 == 0:  # Log every 10 batches
                    wandb.log({
                        "batch": epoch * len(data_loader) + batch_idx,
                        "batch/loss": loss.item(),
                        "batch/token_accuracy": correct.item() / shift_labels.numel()
                    })
                    
                    # Log attention maps from transformer (if available)
                    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                        try:
                            # Select first layer, first head attention map
                            attention_map = outputs.attentions[0][0, 0].cpu().numpy()
                            
                            # Create heatmap
                            plt.figure(figsize=(10, 8))
                            plt.imshow(attention_map, cmap='viridis')
                            plt.colorbar()
                            plt.title(f"Attention Map - Epoch {epoch}, Batch {batch_idx}")
                            
                            # Log to wandb
                            wandb.log({
                                "attention_map": wandb.Image(plt)
                            })
                            plt.close()
                        except Exception as e:
                            print(f"Error plotting attention map: {e}")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate average metrics for the epoch
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        avg_accuracy = token_accuracy / total_tokens if total_tokens > 0 else 0
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Token Accuracy: {avg_accuracy:.4f}")
        
        # Log epoch metrics to wandb
        if log_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/token_accuracy": avg_accuracy,
                "train/learning_rate": optimizer.param_groups[0]['lr']
            })
            
            # Log model parameters histograms (every 5 epochs)
            if epoch % 5 == 0:
                for name, param in graph_llm.named_parameters():
                    if param.requires_grad:
                        wandb.log({f"parameters/{name}": wandb.Histogram(param.data.cpu().numpy())})
        
        # Generate sample graph from fixed prompt (every 5 epochs)
        if epoch % 5 == 0 and text_tokenizer is not None:
            try:
                # Set model to evaluation mode
                graph_llm.eval()
                
                # Generate graph from a fixed prompt for consistent comparison
                sample_prompt = "Generate a small-world network graph"
                
                # Tokenize prompt
                prompt_tokens = text_tokenizer(
                    sample_prompt, 
                    return_tensors="pt"
                ).input_ids.to(device)
                
                # Generate graph tokens
                with torch.no_grad():
                    # Forward pass through LLM to generate tokens
                    output_tokens = graph_llm.base_llm.generate(
                        input_ids=prompt_tokens,
                        max_length=100,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7
                    )
                    
                    # Extract generated tokens after the prompt
                    generated_tokens = output_tokens[0, prompt_tokens.shape[1]:]
                    
                    # Convert to graph tokens and decode
                    Z_list = graph_llm.tokens_to_graph(generated_tokens)
                    adj_matrix, node_features = graph_llm.tokenizer.decode_graph(Z_list)
                    
                    # Visualize the generated graph
                    plt.figure(figsize=(10, 8))
                    G = nx.from_numpy_array(adj_matrix.cpu().numpy())
                    pos = nx.spring_layout(G)
                    nx.draw(G, pos, node_size=50, node_color='blue', alpha=0.8)
                    plt.title(f"Generated Graph - Epoch {epoch}")
                    
                    if log_wandb:
                        wandb.log({
                            f"generated_graph_epoch_{epoch}": wandb.Image(plt)
                        })
                    plt.close()
            except Exception as e:
                print(f"Error generating sample graph: {e}")
                traceback.print_exc()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = graph_llm.state_dict().copy()
            
            # Save checkpoint
            torch.save(best_model_state, "best_graph_llm_checkpoint.pt")
            if log_wandb:
                wandb.save("best_graph_llm_checkpoint.pt")
                wandb.log({"best_epoch": epoch, "best_loss": best_loss})
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 9 or epoch == epochs - 1:
            checkpoint_path = f"graph_llm_checkpoint_epoch_{epoch+1}.pt"
            torch.save(graph_llm.state_dict(), checkpoint_path)
            if log_wandb:
                wandb.save(checkpoint_path)
    
    # Restore best model
    if best_model_state is not None:
        graph_llm.load_state_dict(best_model_state)
    
    # Final model save
    torch.save(graph_llm.state_dict(), "final_graph_llm_checkpoint.pt")
    if log_wandb:
        wandb.save("final_graph_llm_checkpoint.pt")
    
    return graph_llm

# Complete pipeline example

def hierarchical_graphvq_pipeline(use_wandb=True, dataset_name='cora', model_dir='meta-llama/Llama-3.1-8B-Instruct',batch_size=32, epochs=100, \
learning_rate=0.001, train_llm=True, llm_epochs=50, llm_lr=0.0001):
    """
    Pipeline for Hierarchical GraphVQ with both Stage 1 (Graph Tokenizer) and Stage 2 (GraphLLM)
    
    Args:
        use_wandb: Whether to use Weights & Biases for experiment tracking
        dataset_name: Name of the dataset ('cora', 'citeseer', 'pubmed' or from TUDataset)
        batch_size: Batch size for training
        epochs: Number of epochs for training tokenizer
        learning_rate: Learning rate for the tokenizer optimizer
        train_llm: Whether to train the GraphLLM (Stage 2)
        llm_epochs: Number of epochs for training LLM
        llm_lr: Learning rate for the LLM optimizer
    """
    # Define parameters
    input_dim = 1433  # Node feature dimension (will be updated based on dataset)
    hidden_dim = 256  # Hidden layer dimension
    K = 3  # Maximum number of hops
    M_list = [256, 128, 64]  # Codebook sizes for each hop
    p = 20  # Maximum number of neighbors to sample
    num_nodes = 100  # Number of nodes in the graph (will be updated based on dataset)
    tokenizer_epochs = epochs
    tokenizer_lr = learning_rate
    
    if use_wandb:
        # Initialize wandb for the tokenizer training
        wandb.init(
            project="hierarchical-graphvq", 
            name=f"graph_tokenizer_{dataset_name}",
            config={
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "K": K,
                "M_list": M_list,
                "p": p,
                "num_nodes": num_nodes,
                "tokenizer_epochs": tokenizer_epochs,
                "tokenizer_lr": tokenizer_lr,
                "stage": "tokenizer" if not train_llm else "complete",
                "dataset": dataset_name,
                "batch_size": batch_size,
                "train_llm": train_llm,
                "llm_epochs": llm_epochs,
                "llm_lr": llm_lr
            }
        )
    
    # Create models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = f"models/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    try:
        print(f"Loading dataset {dataset_name}...")
        
        # Handle different dataset types
        if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
            # Load a Planetoid dataset (citation networks)
            from torch_geometric.datasets import Planetoid
            from torch_geometric.transforms import NormalizeFeatures
            
            dataset = Planetoid(
                root=f'../data/Cora', 
                name=dataset_name.capitalize(),
                transform=NormalizeFeatures()
            )
            
            # For Planetoid datasets like Cora, there's only one graph, so we need to handle it differently
            data = dataset[0]
            num_nodes = data.num_nodes
            input_dim = dataset.num_features
            
            print(f"Loaded {dataset_name} dataset:")
            print(f"  Number of nodes: {num_nodes}")
            print(f"  Number of edges: {data.edge_index.size(1)}")
            print(f"  Number of node features: {input_dim}")
            print(f"  Number of classes: {dataset.num_classes}")
            
            # For Planetoid datasets, we need to create a custom DataLoader
            # since the dataset contains only one graph
            # We'll create multiple subgraphs for batch processing
            
            # Function to sample subgraphs from the main graph
            def sample_subgraphs(data, num_samples, subgraph_size):
                subgraphs = []
                for _ in range(num_samples):
                    # Randomly select a starting node
                    start_idx = torch.randint(0, data.num_nodes, (1,)).item()
                    
                    # Perform BFS to get connected subgraph
                    visited = set([start_idx])
                    frontier = [start_idx]
                    subgraph_nodes = [start_idx]
                    
                    while len(subgraph_nodes) < subgraph_size and frontier:
                        current = frontier.pop(0)
                        # Get neighbors of current node
                        neighbors = data.edge_index[1][data.edge_index[0] == current].tolist()
                        
                        for neighbor in neighbors:
                            if neighbor not in visited and len(subgraph_nodes) < subgraph_size:
                                visited.add(neighbor)
                                frontier.append(neighbor)
                                subgraph_nodes.append(neighbor)
                    
                    # Create a subgraph with the selected nodes
                    node_idx = torch.tensor(subgraph_nodes)
                    row, col = data.edge_index
                    edge_mask = torch.isin(row, node_idx) & torch.isin(col, node_idx)
                    
                    # Create a new edge_index for the subgraph
                    subgraph_edge_index = data.edge_index[:, edge_mask]
                    
                    # Remap node indices to be consecutive
                    node_idx_map = {int(idx): i for i, idx in enumerate(node_idx)}
                    subgraph_edge_index_mapped = torch.tensor([
                        [node_idx_map[int(idx)] for idx in subgraph_edge_index[0]],
                        [node_idx_map[int(idx)] for idx in subgraph_edge_index[1]]
                    ])
                    
                    # Create a new data object for the subgraph
                    from torch_geometric.data import Data
                    subgraph = Data(
                        x=data.x[node_idx],
                        edge_index=subgraph_edge_index_mapped,
                        y=data.y[node_idx]
                    )
                    
                    subgraphs.append(subgraph)
                
                return subgraphs
            
            # Sample subgraphs for training
            subgraph_size = min(num_nodes, 100)  # Limit subgraph size
            num_samples = 100  # Number of subgraphs to create
            subgraphs = sample_subgraphs(data, num_samples, subgraph_size)
            
            # Create a data loader from the sampled subgraphs
            from torch_geometric.loader import DataLoader
            data_loader = DataLoader(subgraphs, batch_size=batch_size, shuffle=True)
            
            print(f"Created {num_samples} subgraphs of size {subgraph_size} for training")
            
            # Update num_nodes to the subgraph size for the tokenizer
            num_nodes = subgraph_size
            
        else:
            # Try to load a TUDataset
            from torch_geometric.datasets import TUDataset
            from torch_geometric.loader import DataLoader
            
            dataset = TUDataset(root='data/TUDataset', name=dataset_name)
            
            # Get input dimension from dataset if available
            if hasattr(dataset, 'num_features') and dataset.num_features > 0:
                input_dim = dataset.num_features
            
            # Get max number of nodes from the dataset
            max_nodes = 0
            for data in dataset:
                if data.num_nodes > max_nodes:
                    max_nodes = data.num_nodes
            
            # Set num_nodes to the maximum in the dataset, but cap it at 100 for efficiency
            num_nodes = min(max_nodes, 100)
            
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Using input dimension: {input_dim}, num_nodes: {num_nodes}")
        
        # Initialize tokenizer with the correct dimensions
        print("Stage 1: Initializing Graph Tokenizer...")
        tokenizer = GraphTokenizer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            K=K,
            M_list=M_list,
            p=p,
            num_nodes=num_nodes
        )
        
        if use_wandb:
            wandb.config.update({
                "input_dim": input_dim,
                "num_nodes": num_nodes,
                "dataset_name": dataset_name
            })
            
            # Log sample graph visualization
            try:
                if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
                    # Visualize one of the sampled subgraphs
                    sample_data = subgraphs[0]
                else:
                    sample_data = dataset[0]
                
                G = nx.from_edgelist(sample_data.edge_index.t().numpy())
                
                plt.figure(figsize=(10, 10))
                nx.draw(G, node_size=50, with_labels=False)
                plt.title(f"Sample Graph from {dataset_name}")
                
                wandb.log({"sample_graph": wandb.Image(plt)})
                plt.close()  # Close the figure to free memory
            except Exception as e:
                print(f"Error visualizing sample graph: {e}")
                import traceback
                traceback.print_exc()
        
        # Optimizer for tokenizer
        tokenizer_optimizer = optim.Adam(tokenizer.parameters(), lr=tokenizer_lr)
        
        print(f"Training Graph Tokenizer for {tokenizer_epochs} epochs...")
        # Train tokenizer with wandb logging
        tokenizer = train_graph_tokenizer(
            tokenizer, 
            data_loader, 
            tokenizer_optimizer, 
            device, 
            epochs=tokenizer_epochs,
            log_wandb=use_wandb
        )
        
        # Save the trained tokenizer
        tokenizer_path = f"{output_dir}/trained_graph_tokenizer.pt"
        torch.save(tokenizer.state_dict(), tokenizer_path)
        
        if use_wandb:
            wandb.save(tokenizer_path)
            
        print("Graph Tokenizer training completed")
        print(f"Trained tokenizer saved to: {tokenizer_path}")
        
        # Stage 2: Initialize and train Graph LLM (if enabled)
        if train_llm:
            # Close the wandb run for Stage 1
            if use_wandb:
                wandb.finish()
                
                # Start a new wandb run for Stage 2
                wandb.init(
                    project="hierarchical-graphvq", 
                    name=f"graph_llm_{dataset_name}",
                    config={
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dim,
                        "K": K,
                        "M_list": M_list,
                        "dataset": dataset_name,
                        "batch_size": batch_size,
                        "llm_epochs": llm_epochs,
                        "llm_lr": llm_lr,
                        "stage": "llm"
                    }
                )
            
            print("Stage 2: Initializing Graph LLM...")
            try:
                from transformers import AutoModel, AutoTokenizer,AutoConfig
                # Load pretrained model or initialize from scratch
                base_llm_config = AutoConfig.from_pretrained(model_dir)
                base_llm=AutoModel.from_pretrained(model_dir)
                text_tokenizer = AutoTokenizer.from_pretrained(model_dir)
                
                # Create Graph LLM
                graph_llm = GraphLLM(
                    tokenizer=tokenizer,
                    base_llm=base_llm,
                    vocab_size=len(text_tokenizer),
                    embedding_dim=hidden_dim,
                    hidden_dim=hidden_dim
                )
                
                # Add special tokens to text tokenizer
                text_tokenizer = graph_llm.add_special_tokens(text_tokenizer)
                
                # Create a synthetic dataset for text-to-graph generation
                print("Creating synthetic text-graph dataset for LLM training...")
                
                # Create a simple dataset that maps text prompts to graphs
                class TextGraphDataset(torch.utils.data.Dataset):
                    def __init__(self, graph_data, text_tokenizer, num_samples=100, max_length=64):
                        self.graph_data = graph_data
                        self.text_tokenizer = text_tokenizer
                        self.num_samples = num_samples
                        self.max_length = max_length
                        
                        # Create prompts based on dataset type
                        if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
                            # For citation networks, create paper-related prompts
                            self.prompts = [
                                f"Generate a citation network similar to {dataset_name}",
                                f"Create a graph of {dataset_name} paper citations",
                                f"Produce a citation graph in the style of {dataset_name}",
                                f"Generate a network of papers like {dataset_name}",
                                f"Show me a {dataset_name}-like citation structure"
                            ]
                            # Extend prompts by adding numbered variations
                            self.prompts.extend([
                                f"Generate citation graph #{i}" for i in range(num_samples - len(self.prompts))
                            ])
                        else:
                            # For other datasets, create more generic prompts
                            self.prompts = [
                                f"Generate a graph similar to sample {i}" for i in range(num_samples)
                            ]
                        
                    def __len__(self):
                        return self.num_samples
                    
                    def __getitem__(self, idx):
                        # Get text prompt
                        prompt = self.prompts[idx % len(self.prompts)]
                        
                        # Tokenize prompt
                        prompt_tokens = self.text_tokenizer(
                            prompt, 
                            return_tensors="pt", 
                            padding="max_length", 
                            truncation=True,
                            max_length=self.max_length
                        ).input_ids[0]
                        
                        # Get graph - for Planetoid, use a subgraph, for others use the dataset directly
                        if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
                            graph = self.graph_data[idx % len(self.graph_data)]
                        else:
                            # For TUDataset
                            graph = self.graph_data[idx % len(self.graph_data)]
                        
                        return prompt_tokens, graph
                
                # Create text-graph dataset and dataloader
                if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
                    text_graph_dataset = TextGraphDataset(subgraphs, text_tokenizer, num_samples=100)
                else:
                    text_graph_dataset = TextGraphDataset(dataset, text_tokenizer, num_samples=100)
                
                text_graph_dataloader = torch.utils.data.DataLoader(
                    text_graph_dataset, 
                    batch_size=batch_size, 
                    shuffle=True
                )
                
                # Optimizer for LLM
                llm_optimizer = optim.Adam(graph_llm.parameters(), lr=llm_lr)
                
                print(f"Training Graph LLM for {llm_epochs} epochs...")
                # Train LLM with wandb logging
                graph_llm = train_graph_llm(
                    graph_llm, 
                    text_graph_dataloader, 
                    llm_optimizer, 
                    device, 
                    epochs=llm_epochs,
                    log_wandb=use_wandb,
                    text_tokenizer=text_tokenizer
                )
                
                # Save the trained LLM
                llm_path = f"{output_dir}/trained_graph_llm.pt"
                torch.save(graph_llm.state_dict(), llm_path)
                
                if use_wandb:
                    wandb.save(llm_path)
                    
                print("Graph LLM training completed")
                print(f"Trained LLM saved to: {llm_path}")
                
                # Generate a sample graph from text
                try:
                    print("Generating a sample graph from text prompt...")
                    
                    # Set appropriate prompt based on dataset
                    if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
                        prompt = f"Generate a citation network similar to {dataset_name}"
                    else:
                        prompt = f"Generate a graph similar to {dataset_name}"
                    
                    # Move graph_llm to device
                    graph_llm = graph_llm.to(device)
                    
                    # Generate graph
                    adj_matrix, node_features = graph_llm.generate_graph(
                        prompt, 
                        text_tokenizer,
                        max_length=100
                    )
                    
                    # Visualize the generated graph
                    plt.figure(figsize=(10, 10))
                    G = nx.from_numpy_array(adj_matrix.cpu().numpy())
                    pos = nx.spring_layout(G)
                    nx.draw(G, pos, node_size=50, node_color='blue', alpha=0.8)
                    plt.title(f"Generated Graph from Text Prompt")
                    
                    if use_wandb:
                        wandb.log({"generated_graph": wandb.Image(plt)})
                    
                    plt.savefig(f"{output_dir}/generated_graph.png")
                    plt.close()
                    
                    print(f"Generated graph visualization saved to: {output_dir}/generated_graph.png")
                    
                except Exception as e:
                    print(f"Error generating sample graph: {e}")
                    import traceback
                    traceback.print_exc()
                
            except ImportError as e:
                print(f"Error importing transformers library: {e}")
                print("Make sure to install transformers: pip install transformers")
                if use_wandb:
                    wandb.log({"error": str(e)})
            
            except Exception as e:
                print(f"Error during LLM training: {e}")
                import traceback
                traceback.print_exc()
                
                if use_wandb:
                    wandb.log({"error": str(e)})
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        
        if use_wandb:
            wandb.log({"error": str(e)})
    
    print("Hierarchical GraphVQ pipeline completed")
    

if __name__ == "__main__":
    # Add command line arguments for wandb
    import argparse
    parser = argparse.ArgumentParser(description='Hierarchical GraphVQ')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for tracking')
    parser.add_argument('--wandb_key', type=str, help='Weights & Biases API key')
    parser.add_argument('--wandb_project', type=str, default='hierarchical-graphvq', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, help='Weights & Biases entity name')
    parser.add_argument('--model_dir', type=str, default='meta-llama/Llama-3.1-8B-Instruct',help='pretrained or finetuned model location')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dataset', type=str, default='cora', 
                        help='Dataset name (cora, citeseer, pubmed or from TUDataset)')
    parser.add_argument('--train_llm', action='store_true', help='Train the GraphLLM (Stage 2)')
    parser.add_argument('--llm_epochs', type=int, default=50, help='Number of epochs for LLM training')
    parser.add_argument('--llm_lr', type=float, default=0.0001, help='Learning rate for LLM')
    parser.add_argument('--subgraph_size', type=int, default=100, 
                        help='Size of subgraphs to sample from Planetoid datasets')
    parser.add_argument('--num_subgraphs', type=int, default=100, 
                        help='Number of subgraphs to sample from Planetoid datasets')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.use_wandb:
        # Login to wandb if API key is provided
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        
        # Set wandb project and entity
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
    
    hierarchical_graphvq_pipeline(
        use_wandb=args.use_wandb,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        model_dir=args.model_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        train_llm=args.train_llm,
        llm_epochs=args.llm_epochs,
        llm_lr=args.llm_lr
    )
