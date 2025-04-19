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
from models.graph_decoder import *
from models.graph_encoder import *
from models.graph_llm import *
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
        
        for batch_idx, batch in enumerate(data_loader):
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
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
              f"Recon: {avg_recon_loss:.4f}, VQ: {avg_vq_loss:.4f}, Commit: {avg_commit_loss:.4f}")
        
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

def train_graph_llm(graph_llm, data_loader, optimizer, device, epochs=50, log_wandb=True):
    """
    Train the graph LLM
    
    Args:
        graph_llm: Graph LLM model
        data_loader: Data loader for graph-text pairs
        optimizer: Optimizer
        device: Device to run on
        epochs: Number of epochs
        log_wandb: Whether to log metrics to Weights & Biases
    """
    if log_wandb:
        # Initialize wandb run for the LLM
        wandb.init(project="hierarchical-graphvq", name="llm_training", 
                  config={
                      "K": graph_llm.tokenizer.K,
                      "M_list": graph_llm.tokenizer.M_list,
                      "num_graph_tokens": graph_llm.num_graph_tokens,
                      "optimizer": optimizer.__class__.__name__,
                      "learning_rate": optimizer.param_groups[0]['lr'],
                      "epochs": epochs
                  })
    
    graph_llm.to(device)
    graph_llm.train()
    
    for epoch in range(epochs):
        total_loss = 0
        token_accuracy = 0
        total_tokens = 0
        
        for batch_idx, batch in enumerate(data_loader):
            prompt, graph = batch
            prompt = prompt.to(device)
            graph = graph.to(device)
            
            optimizer.zero_grad()
            
            # Get graph tokens
            edge_index, x = graph.edge_index, graph.x
            Z_list = graph_llm.tokenizer.encode_graph(edge_index, x)
            
            # Convert to text tokens
            graph_tokens = graph_llm.graph_to_tokens(Z_list)
            
            # Combine with prompt tokens
            input_tokens = torch.cat([prompt, graph_tokens], dim=1)
            
            # Forward pass
            outputs = graph_llm(input_tokens)
            
            # Calculate loss (next token prediction)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_tokens[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Calculate token prediction accuracy
            with torch.no_grad():
                preds = torch.argmax(shift_logits, dim=-1)
                correct = (preds == shift_labels).float().sum()
                token_accuracy += correct.item()
                total_tokens += shift_labels.numel()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log batch metrics to wandb
            if log_wandb and batch_idx % 10 == 0:  # Log every 10 batches
                wandb.log({
                    "batch": epoch * len(data_loader) + batch_idx,
                    "batch_loss": loss.item(),
                    "batch_token_accuracy": correct.item() / shift_labels.numel(),
                })
                
                # Log attention maps for a few samples (once per epoch)
                if batch_idx == 0 and hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    # Log first layer attention map from first head
                    attention_map = outputs.attentions[0][0, 0].cpu().numpy()
                    wandb.log({
                        "attention_map": wandb.Image(plt.matshow(attention_map)),
                    })
        
        avg_loss = total_loss / len(data_loader)
        avg_accuracy = token_accuracy / total_tokens
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Token Accuracy: {avg_accuracy:.4f}")
        
        # Log epoch metrics to wandb
        if log_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch_loss": avg_loss,
                "epoch_token_accuracy": avg_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            # Generate sample graph from a fixed prompt once per epoch
            if hasattr(graph_llm, 'generate_graph') and epoch % 5 == 0:
                try:
                    # Use a fixed prompt for consistent comparison
                    sample_prompt = "Generate a social network graph with communities"
                    adj_matrix, node_features = graph_llm.generate_graph(sample_prompt, tokenizer=None)
                    
                    # Visualize the generated graph
                    plt.figure(figsize=(10, 10))
                    G = nx.from_numpy_array(adj_matrix.cpu().numpy())
                    pos = nx.spring_layout(G)
                    nx.draw(G, pos, node_size=50, node_color='blue', alpha=0.8)
                    plt.title(f"Generated Graph - Epoch {epoch}")
                    
                    wandb.log({
                        "generated_graph": wandb.Image(plt),
                    })
                except Exception as e:
                    print(f"Error generating sample graph: {e}")
    
    # Save model checkpoint to wandb
    if log_wandb:
        torch.save(graph_llm.state_dict(), "graph_llm_checkpoint.pt")
        wandb.save("graph_llm_checkpoint.pt")
        wandb.finish()
    
    return graph_llm

# Complete pipeline example

def hierarchical_graphvq_pipeline(use_wandb=True, dataset_name='ENZYMES', batch_size=32, epochs=100, learning_rate=0.001):
    """
    Pipeline for Hierarchical GraphVQ that only trains and monitors Stage 1 (Graph Tokenizer)
    
    Args:
        use_wandb: Whether to use Weights & Biases for experiment tracking
        dataset_name: Name of the dataset from TUDataset
        batch_size: Batch size for training
        epochs: Number of epochs for training
        learning_rate: Learning rate for the optimizer
    """
    # Define parameters
    input_dim = 768  # Node feature dimension
    hidden_dim = 256  # Hidden layer dimension
    K = 3  # Maximum number of hops
    M_list = [256, 128, 64]  # Codebook sizes for each hop
    p = 20  # Maximum number of neighbors to sample
    num_nodes = 100  # Number of nodes in the graph
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
                "stage": "tokenizer_only",
                "dataset": dataset_name,
                "batch_size": batch_size
            }
        )
    
    # Create models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Stage 1: Initialize and train Graph Tokenizer
    print("Stage 1: Initializing Graph Tokenizer...")
    tokenizer = GraphTokenizer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        K=K,
        M_list=M_list,
        p=p,
        num_nodes=num_nodes
    )
    # TODO: Apply stage 1 result here
    
    # Load your graph dataset and create data loader
    try:
        print(f"Loading dataset {dataset_name}...")
        from torch_geometric.datasets import TUDataset
        from torch_geometric.loader import DataLoader
        
        # Try to load a graph dataset
        #dataset = TUDataset(root='data/TUDataset', name=dataset_name)
        
        from torch_geometric.datasets import Planetoid
    
        dataset = Planetoid(root='./data/Cora', name='Cora')
        data = dataset[0]
        print("加载数据集完成")
        
        # TODO: use the feature as stage 1
        # 获取节点特征
        node_features = data.x.numpy()
        num_nodes = node_features.shape[0]
        
        # 获取节点标签
        labels = data.y.numpy()
        
        # 论文类别名称
        paper_categories = ['理论', '强化学习', '遗传算法', '神经网络', '概率方法', '规则学习', '案例推理']
        
        # 将图转换为NetworkX格式
        G = to_networkx(data)
        
        # 计算结构嵌入的维度
        structure_dim = int((args.sample_size**(args.use_hop+1)-1)/(args.sample_size-1))
        print(f"结构嵌入维度: {structure_dim}")
        
        # 准备节点嵌入和描述
        node_embeddings = {}
        
        print("正在处理节点嵌入和描述...")
        for i in range(num_nodes):
            # 获取结构嵌入
            structure_embedding = process_graph_structure(
                G, i, args.use_hop, args.sample_size
            )
            
            # 组合嵌入
            combined_embedding = combine_embeddings(
                node_features[i], structure_embedding, args.embedding_type
            )
        
        node_embeddings[f"node{i}"] = combined_embedding
        
        # Get input dimension from dataset if available
        if hasattr(dataset, 'num_features') and dataset.num_features > 0:
            input_dim = dataset.num_features
            print(f"Using dataset feature dimension: {input_dim}")
            
            # Reinitialize tokenizer with correct input dimension
            tokenizer = GraphTokenizer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                K=K,
                M_list=M_list,
                p=p,
                num_nodes=num_nodes
            )
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if use_wandb:
            wandb.log({
                "dataset_name": dataset_name, 
                "dataset_size": len(dataset),
                "input_dim": input_dim
            })
            
            # Log sample graph visualization
            try:
                sample_data = dataset[0]
                G = nx.from_edgelist(sample_data.edge_index.t().numpy())
                
                plt.figure(figsize=(10, 10))
                nx.draw(G, node_size=50, with_labels=False)
                plt.title(f"Sample Graph from {dataset_name}")
                
                wandb.log({"sample_graph": wandb.Image(plt)})
                plt.close()  # Close the figure to free memory
            except Exception as e:
                print(f"Error visualizing sample graph: {e}")
        
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
        output_dir = f"models/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = f"{output_dir}/trained_graph_tokenizer.pt"
        torch.save(tokenizer.state_dict(), model_path)
        
        if use_wandb:
            wandb.save(model_path)
            
        print("Graph Tokenizer training completed")
        print(f"Trained model saved to: {model_path}")
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        
        if use_wandb:
            wandb.log({"error": str(e)})
    
    print("Hierarchical GraphVQ Stage 1 pipeline completed")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    # Add command line arguments for wandb

    parser = argparse.ArgumentParser(description='Hierarchical GraphVQ')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for tracking')
    parser.add_argument('--wandb_key', type=str, help='Weights & Biases API key')
    parser.add_argument('--wandb_project', type=str, default='hierarchical-graphvq', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, help='Weights & Biases entity name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    
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
        epochs=args.epochs,
        learning_rate=args.lr
    )

if __name__ == "__main__":
    hierarchical_graphvq_pipeline()