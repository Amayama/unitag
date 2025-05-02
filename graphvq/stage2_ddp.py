import os
import argparse
import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Planetoid
import datetime
from transformers import AutoModel, AutoTokenizer, AutoConfig
import wandb
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import sys
import traceback

# Import model definitions
from models.graph_decoder import GraphDecoder
from models.graph_encoder import GNNEncoder
from models.graph_llm import GraphLLM
from models.graph_tokenizer import GraphTokenizer
from models.text_graph_dataset import TextGraphDataset


def setup_distributed(args):
    """
    Initialize the distributed environment with more debug info
    """
    # Print debug info
    print(f"Process {args.local_rank}: Starting setup_distributed")
    print(f"Process {args.local_rank}: CUDA available: {torch.cuda.is_available()}")
    print(f"Process {args.local_rank}: GPU count: {torch.cuda.device_count()}")
    
    if args.local_rank == -1:
        # Single GPU code...
        print("Running in single GPU mode")
    else:
        try:
            print(f"Process {args.local_rank}: Setting device")
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            
            print(f"Process {args.local_rank}: Initializing process group")
            # Add timeout to prevent indefinite hanging
            dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
            
            print(f"Process {args.local_rank}: Process group initialized")
            args.world_size = dist.get_world_size()
            args.n_gpu = 1
            args.is_master = args.local_rank == 0
        except Exception as e:
            print(f"Process {args.local_rank}: Error in setup: {str(e)}")
            raise
    
    args.device = device
    print(f"Process {args.local_rank}: setup_distributed completed")
    return args

def cleanup_distributed():
    """
    Clean up the distributed environment
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def set_seed(seed):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_graph_tokenizer_ddp(tokenizer, data_loader, optimizer, args, epochs=100):
    """
    Train the graph tokenizer with Distributed Data Parallel
    """
    # Setup wandb (only on master process)
    if args.use_wandb and args.is_master:
        wandb.init(
            project="graphvq-ddp",
            name=f"tokenizer_ddp_{args.dataset}",
            config={
                "model": "GraphTokenizer",
                "epochs": epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "world_size": args.world_size,
                "input_dim": tokenizer.input_dim,
                "hidden_dim": tokenizer.hidden_dim,
                "K": tokenizer.K,
                "M_list": tokenizer.M_list,
                "p": tokenizer.p,
            }
        )
    
    # Move tokenizer to correct device
    tokenizer = tokenizer.to(args.device)
    
    # Wrap model with DDP (only if distributed training is enabled)
    if args.local_rank != -1:
        tokenizer = DDP(
            tokenizer, 
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=args.is_master
    )
    
    best_loss = float('inf')
    best_model_state = None
    
    # Define model directory for saving checkpoints
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Set epoch for data sampler to ensure proper shuffling
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)
        
        # Training mode
        if isinstance(tokenizer, DDP):
            tokenizer.module.train()
        else:
            tokenizer.train()
        
        total_loss = 0
        recon_loss_sum = 0
        vq_loss_sum = 0
        commit_loss_sum = 0
        recon_x_loss_sum = 0
        recon_adj_loss_sum = 0
        batch_count = 0
        
        # Create progress bar (only on master process)
        if args.is_master:
            progress = tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(args.device)
            edge_index, x = batch.edge_index, batch.x
            
            optimizer.zero_grad()
            
            # Forward pass through tokenizer
            if isinstance(tokenizer, DDP):
                loss, E, Z = tokenizer.module.train_tokenizer(edge_index, x)
                
                # Calculate loss components for logging
                with torch.no_grad():
                    _, loss_components = tokenizer.module.calculate_loss(
                        edge_index=edge_index,
                        X=x,
                        E=E,
                        assignments=[(h_k, quantized, _) for h_k, quantized, _ in tokenizer.module.last_assignments]
                    )
            else:
                loss, E, Z = tokenizer.train_tokenizer(edge_index, x)
                
                # Calculate loss components for logging
                with torch.no_grad():
                    _, loss_components = tokenizer.calculate_loss(
                        edge_index=edge_index,
                        X=x,
                        E=E,
                        assignments=[(h_k, quantized, _) for h_k, quantized, _ in tokenizer.last_assignments]
                    )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            batch_count += 1
            total_loss += loss_components['total_loss']
            recon_loss_sum += loss_components['recon_loss']
            vq_loss_sum += loss_components['vq_loss']
            commit_loss_sum += loss_components['commit_loss']
            recon_x_loss_sum += loss_components['recon_x_loss']
            recon_adj_loss_sum += loss_components['recon_adj_loss']
            
            # Update progress bar on master process
            if args.is_master:
                progress.update(1)
                progress.set_postfix({
                    'loss': loss_components['total_loss'],
                    'recon': loss_components['recon_loss']
                })
                
                # Log batch metrics to wandb (every 10 batches)
                if args.use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        "batch": epoch * len(data_loader) + batch_idx,
                        "batch/total_loss": loss_components['total_loss'],
                        "batch/recon_loss": loss_components['recon_loss'],
                        "batch/vq_loss": loss_components['vq_loss'],
                        "batch/commit_loss": loss_components['commit_loss'],
                        "learning_rate": optimizer.param_groups[0]['lr']
                    })
        
        # Close progress bar
        if args.is_master:
            progress.close()
        
        # Average the losses across processes
        if args.local_rank != -1:
            # Create tensors for gathering
            world_size = dist.get_world_size()
            total_loss_tensor = torch.tensor([total_loss, batch_count], dtype=torch.float32, device=args.device)
            recon_loss_tensor = torch.tensor([recon_loss_sum], dtype=torch.float32, device=args.device)
            vq_loss_tensor = torch.tensor([vq_loss_sum], dtype=torch.float32, device=args.device)
            commit_loss_tensor = torch.tensor([commit_loss_sum], dtype=torch.float32, device=args.device)
            
            # All-reduce the tensors
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(recon_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(vq_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(commit_loss_tensor, op=dist.ReduceOp.SUM)
            
            # Calculate global averages
            global_total_loss = total_loss_tensor[0] / (total_loss_tensor[1] * world_size)
            global_recon_loss = recon_loss_tensor[0] / (total_loss_tensor[1] * world_size)
            global_vq_loss = vq_loss_tensor[0] / (total_loss_tensor[1] * world_size)
            global_commit_loss = commit_loss_tensor[0] / (total_loss_tensor[1] * world_size)
        else:
            # Single-GPU case
            global_total_loss = total_loss / batch_count
            global_recon_loss = recon_loss_sum / batch_count
            global_vq_loss = vq_loss_sum / batch_count
            global_commit_loss = commit_loss_sum / batch_count
        
        # Update learning rate scheduler
        scheduler.step(global_total_loss)
        
        # Print epoch summary (master process only)
        if args.is_master:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {global_total_loss:.4f}, "
                  f"Recon: {global_recon_loss:.4f}, VQ: {global_vq_loss:.4f}, "
                  f"Commit: {global_commit_loss:.4f}")
            
            # Log epoch metrics to wandb
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": global_total_loss,
                    "train/recon_loss": global_recon_loss,
                    "train/vq_loss": global_vq_loss,
                    "train/commit_loss": global_commit_loss,
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                })
            
            # Save model checkpoint if it's the best so far
            if global_total_loss < best_loss:
                best_loss = global_total_loss
                
                # Get model state dict (DDP unwrapping)
                if isinstance(tokenizer, DDP):
                    best_model_state = tokenizer.module.state_dict()
                else:
                    best_model_state = tokenizer.state_dict()
                
                # Save best model checkpoint
                best_model_path = os.path.join(args.output_dir, "best_tokenizer.pt")
                torch.save(best_model_state, best_model_path)
                
                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.save(best_model_path)
        
        # Save regular checkpoints every 10 epochs
        if args.is_master and (epoch + 1) % 10 == 0:
            # Get model state dict
            if isinstance(tokenizer, DDP):
                checkpoint_state = tokenizer.module.state_dict()
            else:
                checkpoint_state = tokenizer.state_dict()
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"tokenizer_epoch_{epoch+1}.pt")
            torch.save(checkpoint_state, checkpoint_path)
            
            if args.use_wandb:
                wandb.save(checkpoint_path)
        
        # Make sure all processes are synchronized
        if args.local_rank != -1:
            dist.barrier()
    
    # Save final model (master process only)
    if args.is_master:
        # Save the final model
        if isinstance(tokenizer, DDP):
            final_state = tokenizer.module.state_dict()
        else:
            final_state = tokenizer.state_dict()
        
        final_path = os.path.join(args.output_dir, "final_tokenizer.pt")
        torch.save(final_state, final_path)
        
        if args.use_wandb:
            wandb.save(final_path)
            wandb.finish()
    
    # Load best model on all processes
    if best_model_state is not None:
        if args.local_rank != -1:
            # Wait for master to save the best model
            dist.barrier()
        
        # Load the best model
        best_model_path = os.path.join(args.output_dir, "best_tokenizer.pt")
        if os.path.exists(best_model_path):
            if isinstance(tokenizer, DDP):
                tokenizer.module.load_state_dict(torch.load(best_model_path, map_location=args.device))
            else:
                tokenizer.load_state_dict(torch.load(best_model_path, map_location=args.device))
    
    return tokenizer

def train_graph_llm_ddp(graph_llm, data_loader, optimizer, args, text_tokenizer=None, epochs=50):
    """
    Train the graph LLM with Distributed Data Parallel
    """
    # Setup wandb (only on master process)
    if args.use_wandb and args.is_master:
        wandb.init(
            project="graphvq-ddp",
            name=f"llm_ddp_{args.dataset}",
            config={
                "model": "GraphLLM",
                "epochs": epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.llm_lr,
                "world_size": args.world_size,
                "base_llm": args.model_dir,
            }
        )
    
    # Move model to correct device
    graph_llm = graph_llm.to(args.device)
    
    # Wrap model with DDP (only if distributed training is enabled)
    if args.local_rank != -1:
        graph_llm = DDP(
            graph_llm, 
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=args.is_master
    )
    
    best_loss = float('inf')
    best_model_state = None
    
    # Define model directory for saving checkpoints
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Set epoch for data sampler to ensure proper shuffling
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)
        
        # Training mode
        if isinstance(graph_llm, DDP):
            graph_llm.module.train()
        else:
            graph_llm.train()
        
        total_loss = 0
        token_accuracy = 0
        total_tokens = 0
        batch_count = 0
        
        # Create progress bar (only on master process)
        if args.is_master:
            progress = tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(data_loader):
            try:
                # Unpack batch
                prompt_tokens, graph = batch
                
                # Move data to device
                prompt_tokens = prompt_tokens.to(args.device)
                graph = graph.to(args.device)
                
                optimizer.zero_grad()
                
                # Get graph tokens from the tokenizer
                if isinstance(graph_llm, DDP):
                    edge_index, x = graph.edge_index, graph.x
                    Z_list = graph_llm.module.tokenizer.encode_graph(edge_index, x)
                    
                    # Convert to text tokens
                    graph_tokens = graph_llm.module.graph_to_tokens(Z_list)
                else:
                    edge_index, x = graph.edge_index, graph.x
                    Z_list = graph_llm.tokenizer.encode_graph(edge_index, x)
                    
                    # Convert to text tokens
                    graph_tokens = graph_llm.graph_to_tokens(Z_list)
                
                # Reshape tokens if needed
                if len(prompt_tokens.shape) == 1:
                    prompt_tokens = prompt_tokens.unsqueeze(0)
                if len(graph_tokens.shape) == 1:
                    graph_tokens = graph_tokens.unsqueeze(0)
                
                # Match batch dimensions
                batch_size = min(prompt_tokens.size(0), graph_tokens.size(0))
                prompt_tokens = prompt_tokens[:batch_size]
                graph_tokens = graph_tokens[:batch_size]
                
                # Combine tokens
                input_tokens = torch.cat([prompt_tokens, graph_tokens], dim=1)
                
                # Forward pass
                outputs = graph_llm(input_tokens)
                
                # Calculate loss
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = input_tokens[..., 1:].contiguous()
                
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Calculate accuracy
                with torch.no_grad():
                    preds = torch.argmax(shift_logits, dim=-1)
                    correct = (preds == shift_labels).float().sum()
                    token_accuracy += correct.item()
                    total_tokens += shift_labels.numel()
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(graph_llm.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update metrics
                batch_count += 1
                total_loss += loss.item()
                
                # Update progress bar on master process
                if args.is_master:
                    progress.update(1)
                    acc = correct.item() / shift_labels.numel() if shift_labels.numel() > 0 else 0
                    progress.set_postfix({
                        'loss': loss.item(),
                        'acc': f"{acc:.4f}"
                    })
                    
                    # Log batch metrics to wandb (every 10 batches)
                    if args.use_wandb and batch_idx % 10 == 0:
                        wandb.log({
                            "batch": epoch * len(data_loader) + batch_idx,
                            "batch/loss": loss.item(),
                            "batch/token_accuracy": acc,
                            "learning_rate": optimizer.param_groups[0]['lr']
                        })
            
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                traceback.print_exc()
                continue
        
        # Close progress bar
        if args.is_master:
            progress.close()
        
        # Average the metrics across processes
        if args.local_rank != -1:
            # Create tensors for gathering
            world_size = dist.get_world_size()
            metrics_tensor = torch.tensor(
                [total_loss, token_accuracy, total_tokens, batch_count], 
                dtype=torch.float32, device=args.device
            )
            
            # All-reduce the tensor
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            
            # Extract global metrics
            global_total_loss = metrics_tensor[0]
            global_token_accuracy = metrics_tensor[1]
            global_total_tokens = metrics_tensor[2]
            global_batch_count = metrics_tensor[3]
        else:
            # Single-GPU case
            global_total_loss = total_loss
            global_token_accuracy = token_accuracy
            global_total_tokens = total_tokens
            global_batch_count = batch_count
        
        # Calculate average metrics
        avg_loss = global_total_loss / global_batch_count if global_batch_count > 0 else 0
        avg_accuracy = global_token_accuracy / global_total_tokens if global_total_tokens > 0 else 0
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Print epoch summary (master process only)
        if args.is_master:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"Token Accuracy: {avg_accuracy:.4f}")
            
            # Log epoch metrics to wandb
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": avg_loss,
                    "train/token_accuracy": avg_accuracy,
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                })
            
            # Generate sample graph for visualization (every 5 epochs)
            if (epoch + 1) % 5 == 0 and text_tokenizer is not None:
                try:
                    # Set model to evaluation mode
                    if isinstance(graph_llm, DDP):
                        graph_llm.module.eval()
                    else:
                        graph_llm.eval()
                    
                    # Generate graph from prompt
                    sample_prompt = f"Generate a graph similar to {args.dataset}"
                    
                    # Tokenize prompt
                    prompt_tokens = text_tokenizer(
                        sample_prompt, 
                        return_tensors="pt"
                    ).input_ids.to(args.device)
                    
                    # Generate graph
                    with torch.no_grad():
                        # Get LLM model (unwrap DDP if needed)
                        model = graph_llm.module if isinstance(graph_llm, DDP) else graph_llm
                        
                        # Forward pass to generate tokens
                        output_tokens = model.base_llm.generate(
                            input_ids=prompt_tokens,
                            max_length=150,
                            do_sample=True,
                            top_p=0.9,
                            temperature=0.7
                        )
                        
                        # Extract generated tokens
                        generated_tokens = output_tokens[0, prompt_tokens.shape[1]:]
                        
                        # Convert to graph tokens and decode
                        Z_list = model.tokens_to_graph(generated_tokens)
                        adj_matrix, node_features = model.tokenizer.decode_graph(Z_list)
                        
                        # Visualize the graph
                        plt.figure(figsize=(10, 8))
                        G = nx.from_numpy_array(adj_matrix.cpu().numpy())
                        pos = nx.spring_layout(G)
                        nx.draw(G, pos, node_size=50, node_color='blue', alpha=0.8)
                        plt.title(f"Generated Graph - Epoch {epoch+1}")
                        
                        # Save figure
                        sample_path = os.path.join(args.output_dir, f"sample_graph_epoch_{epoch+1}.png")
                        plt.savefig(sample_path)
                        
                        if args.use_wandb:
                            wandb.log({f"sample_graph_epoch_{epoch+1}": wandb.Image(plt)})
                        
                        plt.close()
                
                except Exception as e:
                    print(f"Error generating sample: {e}")
                    traceback.print_exc()
            
            # Save model checkpoint if it's the best so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                
                # Get model state dict (DDP unwrapping)
                if isinstance(graph_llm, DDP):
                    best_model_state = graph_llm.module.state_dict()
                else:
                    best_model_state = graph_llm.state_dict()
                
                # Save best model checkpoint
                best_model_path = os.path.join(args.output_dir, "best_graph_llm.pt")
                torch.save(best_model_state, best_model_path)
                
                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.save(best_model_path)
        
        # Save regular checkpoints every 10 epochs
        if args.is_master and (epoch + 1) % 10 == 0:
            # Get model state dict
            if isinstance(graph_llm, DDP):
                checkpoint_state = graph_llm.module.state_dict()
            else:
                checkpoint_state = graph_llm.state_dict()
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"graph_llm_epoch_{epoch+1}.pt")
            torch.save(checkpoint_state, checkpoint_path)
            
            if args.use_wandb:
                wandb.save(checkpoint_path)
        
        # Make sure all processes are synchronized
        if args.local_rank != -1:
            dist.barrier()
    
    # Save final model (master process only)
    if args.is_master:
        # Save the final model
        if isinstance(graph_llm, DDP):
            final_state = graph_llm.module.state_dict()
        else:
            final_state = graph_llm.state_dict()
        
        final_path = os.path.join(args.output_dir, "final_graph_llm.pt")
        torch.save(final_state, final_path)
        
        if args.use_wandb:
            wandb.save(final_path)
            wandb.finish()
    
    # Load best model on all processes
    if best_model_state is not None:
        if args.local_rank != -1:
            # Wait for master to save the best model
            dist.barrier()
        
        # Load the best model
        best_model_path = os.path.join(args.output_dir, "best_graph_llm.pt")
        if os.path.exists(best_model_path):
            if isinstance(graph_llm, DDP):
                graph_llm.module.load_state_dict(torch.load(best_model_path, map_location=args.device))
            else:
                graph_llm.load_state_dict(torch.load(best_model_path, map_location=args.device))
    
    return graph_llm

def hierarchical_graphvq_pipeline_ddp(args):
    """
    DDP version of the Hierarchical GraphVQ pipeline
    """
    # Set random seeds for reproducibility
    set_seed(args.seed)
    
    # Initialize distributed training
    args = setup_distributed(args)
    
    # Define model parameters
    input_dim = 1433  # Will be updated based on dataset
    hidden_dim = 256
    K = 3
    M_list = [256, 128, 64]
    p = 20
    num_nodes = 100  # Will be updated based on dataset
    
    if args.is_master:
        print(f"Starting Hierarchical GraphVQ training with {args.world_size} GPUs")
        print(f"Using device: {args.device}, local_rank: {args.local_rank}")
        print(f"Dataset: {args.dataset}, Batch size: {args.batch_size}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Make sure all processes are synchronized
    if args.local_rank != -1:
        dist.barrier()
    
    # Load dataset
    try:
        if args.is_master:
            print(f"Loading dataset {args.dataset}...")
        
        if args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
            # Load a Planetoid dataset
            from torch_geometric.transforms import NormalizeFeatures
            
            dataset = Planetoid(
                root=f'../data/{args.dataset}', 
                name=args.dataset.capitalize(),
                transform=NormalizeFeatures()
            )
            
            # Extract dataset properties
            data = dataset[0]
            num_nodes = data.num_nodes
            input_dim = dataset.num_features
            
            if args.is_master:
                print(f"Loaded {args.dataset} dataset:")
                print(f"  Number of nodes: {num_nodes}")
                print(f"  Number of edges: {data.edge_index.size(1)}")
                print(f"  Number of node features: {input_dim}")
                print(f"  Number of classes: {dataset.num_classes}")
            
            # Function to sample subgraphs for training
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
            subgraph_size = min(num_nodes, args.subgraph_size)
            num_samples = args.num_subgraphs
            subgraphs = sample_subgraphs(data, num_samples, subgraph_size)
            
            # Create data loader with distributed sampler
            if args.local_rank != -1:
                train_sampler = DistributedSampler(
                    subgraphs, 
                    num_replicas=args.world_size,
                    rank=args.local_rank
                )
            else:
                train_sampler = None
            
            data_loader = DataLoader(
                subgraphs, 
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                drop_last=True
            )
            
            if args.is_master:
                print(f"Created {num_samples} subgraphs of size {subgraph_size}")
                print(f"Each process will see ~{len(data_loader) * args.batch_size} samples per epoch")
            
            # Update num_nodes to the subgraph size for the tokenizer
            num_nodes = subgraph_size
            
        else:
            # Try to load a TUDataset
            from torch_geometric.datasets import TUDataset
            
            dataset = TUDataset(root='./data/TUDataset', name=args.dataset)
            
            # Get input dimension from dataset
            if hasattr(dataset, 'num_features') and dataset.num_features > 0:
                input_dim = dataset.num_features
            
            # Get max number of nodes from the dataset
            max_nodes = 0
            for data in dataset:
                if data.num_nodes > max_nodes:
                    max_nodes = data.num_nodes
            
            # Set num_nodes to the maximum in the dataset, but cap it
            num_nodes = min(max_nodes, 100)
            
            # Create data loader with distributed sampler
            if args.local_rank != -1:
                train_sampler = DistributedSampler(
                    dataset, 
                    num_replicas=args.world_size,
                    rank=args.local_rank
                )
            else:
                train_sampler = None
            
            data_loader = DataLoader(
                dataset, 
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                drop_last=True
            )
            
            if args.is_master:
                print(f"Loaded TUDataset {args.dataset}:")
                print(f"  Number of graphs: {len(dataset)}")
                print(f"  Max number of nodes: {max_nodes}")
                print(f"  Input dimension: {input_dim}")
                print(f"Each process will see ~{len(data_loader) * args.batch_size} samples per epoch")
        
        # Initialize tokenizer with the correct dimensions
        if args.is_master:
            print("Stage 1: Initializing Graph Tokenizer...")
        
        tokenizer = GraphTokenizer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            K=K,
            M_list=M_list,
            p=p,
            num_nodes=num_nodes
        )
        
        # Create optimizer for tokenizer
        tokenizer_optimizer = optim.Adam(tokenizer.parameters(), lr=args.lr)
        
        # Train tokenizer with DDP
        if args.is_master:
            print(f"Training Graph Tokenizer for {args.epochs} epochs...")
        
        tokenizer = train_graph_tokenizer_ddp(
            tokenizer,
            data_loader,
            tokenizer_optimizer,
            args,
            epochs=args.epochs
        )
        
        if args.is_master:
            print("Graph Tokenizer training completed")
        
        # Make sure all processes are synchronized
        if args.local_rank != -1:
            dist.barrier()
        
        # Stage 2: Train Graph LLM (if enabled)
        if args.train_llm:
            if args.is_master:
                print("Stage 2: Initializing Graph LLM...")
            
            try:
                
                # Load pretrained model
                base_llm_config = AutoConfig.from_pretrained(args.model_dir)
                base_llm = AutoModel.from_pretrained(args.model_dir)
                text_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
                
                # Ensure padding token exists
                if text_tokenizer.pad_token is None:
                    text_tokenizer.pad_token = text_tokenizer.eos_token
                
                # Create Graph LLM
                if isinstance(tokenizer, DDP):
                    graph_tokenizer = tokenizer.module
                else:
                    graph_tokenizer = tokenizer
                
                graph_llm = GraphLLM(
                    tokenizer=graph_tokenizer,
                    base_llm=base_llm,
                    vocab_size=len(text_tokenizer),
                    embedding_dim=hidden_dim,
                    hidden_dim=hidden_dim
                )
                
                # Add special tokens to text tokenizer
                if args.is_master:
                    print("Adding special tokens to text tokenizer...")
                
                text_tokenizer = graph_llm.add_special_tokens(text_tokenizer)
                
                # Create a synthetic dataset for text-to-graph generation
                if args.is_master:
                    print("Creating synthetic text-graph dataset for LLM training...")
                
                # Create text-graph dataset
                
                
                # Create text-graph dataset and dataloader
                if args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
                    text_graph_dataset = TextGraphDataset(subgraphs, text_tokenizer, num_samples=100)
                else:
                    text_graph_dataset = TextGraphDataset(dataset, text_tokenizer, num_samples=100)
                
                # Create distributed sampler for LLM training
                if args.local_rank != -1:
                    llm_sampler = DistributedSampler(
                        text_graph_dataset, 
                        num_replicas=args.world_size,
                        rank=args.local_rank
                    )
                else:
                    llm_sampler = None
                
                text_graph_dataloader = torch.utils.data.DataLoader(
                    text_graph_dataset, 
                    batch_size=args.batch_size,
                    shuffle=(llm_sampler is None),
                    sampler=llm_sampler,
                    drop_last=True
                )
                
                # Create optimizer for LLM
                llm_optimizer = optim.Adam(graph_llm.parameters(), lr=args.llm_lr)
                
                # Train LLM with DDP
                if args.is_master:
                    print(f"Training Graph LLM for {args.llm_epochs} epochs...")
                
                graph_llm = train_graph_llm_ddp(
                    graph_llm,
                    text_graph_dataloader,
                    llm_optimizer,
                    args,
                    text_tokenizer=text_tokenizer,
                    epochs=args.llm_epochs
                )
                
                if args.is_master:
                    print("Graph LLM training completed")
                
            except ImportError as e:
                if args.is_master:
                    print(f"Error importing transformers library: {e}")
                    print("Make sure to install transformers: pip install transformers")
            
            except Exception as e:
                if args.is_master:
                    print(f"Error during LLM training: {e}")
                    traceback.print_exc()
        
    except Exception as e:
        if args.is_master:
            print(f"Error during pipeline execution: {e}")
            traceback.print_exc()
    
    # Clean up distributed environment
    cleanup_distributed()
    
    if args.is_master:
        print("Hierarchical GraphVQ DDP pipeline completed")

def main():
    parser = argparse.ArgumentParser(description='Hierarchical GraphVQ with DDP')
    
    # Distributed training arguments
    parser.add_argument('--local_rank', "--local-rank", type=int, default=-1, 
                        help='Local rank for distributed training (-1 for single GPU)')
    
    # Dataset and training arguments
    parser.add_argument('--dataset', type=str, default='cora', 
                        help='Dataset name (cora, citeseer, pubmed or TUDataset)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Per-GPU batch size')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs for tokenizer training')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate for tokenizer')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./models/ddp', 
                        help='Output directory for model checkpoints')
    
    # Subgraph sampling arguments
    parser.add_argument('--subgraph_size', type=int, default=100, 
                        help='Size of subgraphs to sample from Planetoid datasets')
    parser.add_argument('--num_subgraphs', type=int, default=100, 
                        help='Number of subgraphs to sample from Planetoid datasets')
    
    # LLM training arguments
    parser.add_argument('--train_llm', action='store_true', 
                        help='Train the GraphLLM (Stage 2)')
    parser.add_argument('--llm_epochs', type=int, default=30, 
                        help='Number of epochs for LLM training')
    parser.add_argument('--llm_lr', type=float, default=0.0001, 
                        help='Learning rate for LLM')
    parser.add_argument('--model_dir', type=str, default='meta-llama/Llama-3.1-8B-Instruct', 
                        help='Pretrained LLM model directory')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true', 
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='graphvq-ddp', 
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, 
                        help='Wandb entity name')
    
    args = parser.parse_args()
    
    # Set environment variables for DDP

    # Run the pipeline
    hierarchical_graphvq_pipeline_ddp(args)

if __name__ == "__main__":
    main()