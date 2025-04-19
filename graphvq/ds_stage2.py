def create_deepspeed_config(
    stage=2,  # ZeRO optimization stage (0, 1, 2, 3)
    offload_optimizer=False,  # Offload optimizer to CPU
    offload_parameters=False,  # Offload parameters to CPU
    gradient_accumulation_steps=1,
    fp16=True,  # Use fp16 for training
    learning_rate=5e-5,
    warmup_steps=100,
    gradient_clipping=1.0,
    train_batch_size=4,  # Per-GPU batch size
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8
):
    """
    Create DeepSpeed configuration
    
    Args:
        stage: ZeRO optimization stage
        offload_optimizer: Whether to offload optimizer states to CPU
        offload_parameters: Whether to offload parameters to CPU
        gradient_accumulation_steps: Number of gradient accumulation steps
        fp16: Whether to use fp16 for training
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        gradient_clipping: Gradient clipping threshold
        train_batch_size: Per-GPU batch size
        adam_beta1: Adam beta1 parameter
        adam_beta2: Adam beta2 parameter
        adam_epsilon: Adam epsilon parameter
        
    Returns:
        DeepSpeed configuration dictionary
    """
    config = {
        "train_batch_size": train_batch_size * torch.cuda.device_count() * gradient_accumulation_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": learning_rate,
                "betas": [adam_beta1, adam_beta2],
                "eps": adam_epsilon
            }
        },
        
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": warmup_steps
            }
        },
        
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {
                "device": "cpu" if offload_optimizer else "none"
            },
            "offload_param": {
                "device": "cpu" if offload_parameters else "none"
            },
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8
        }
    }
    
    if fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,  # Dynamic loss scaling
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    
    return config

# Dataset for graph-text pairs
class GraphTextDataset(torch.utils.data.Dataset):
    """
    Dataset for graph-text pairs for training the GraphLLM
    """
    def __init__(self, text_tokenizer, graph_tokenizer, graph_data, text_prompts, max_length=512):
        """
        Initialize the dataset
        
        Args:
            text_tokenizer: Text tokenizer
            graph_tokenizer: Graph tokenizer
            graph_data: List of graph data (edge_index, x)
            text_prompts: List of text prompts
            max_length: Maximum sequence length
        """
        self.text_tokenizer = text_tokenizer
        self.graph_tokenizer = graph_tokenizer
        self.graph_data = graph_data
        self.text_prompts = text_prompts
        self.max_length = max_length
        
        assert len(graph_data) == len(text_prompts), "Number of graphs and prompts must match"
    
    def __len__(self):
        return len(self.graph_data)
    
    def __getitem__(self, idx):
        # Get graph data
        edge_index, x = self.graph_data[idx]
        
        # Encode graph into tokens
        graph_token_indices = self.graph_tokenizer.encode_graph(edge_index, x)
        
        # Get text prompt
        prompt = self.text_prompts[idx]
        
        # Tokenize text prompt
        prompt_tokens = self.text_tokenizer(
            prompt, 
            max_length=self.max_length // 2,  # Reserve half for graph tokens
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "prompt_tokens": prompt_tokens,
            "graph_token_indices": graph_token_indices,
            "edge_index": edge_index,
            "x": x,
            "prompt_text": prompt
        }

# Collate function for batching
def graph_text_collate_fn(batch, graph_llm):
    """
    Collate function for batching graph-text pairs
    
    Args:
        batch: Batch of graph-text pairs
        graph_llm: GraphLLM model for tokenization
        
    Returns:
        Batched inputs for training
    """
    # Extract prompt tokens
    prompt_tokens = {
        "input_ids": torch.cat([item["prompt_tokens"]["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.cat([item["prompt_tokens"]["attention_mask"] for item in batch], dim=0)
    }
    
    # Convert graph tokens to text tokens
    graph_text_tokens = []
    for item in batch:
        graph_tokens = graph_llm.graph_to_tokens(item["graph_token_indices"])
        graph_text_tokens.append(graph_tokens)
    
    # Concatenate graph tokens
    graph_text_tokens = torch.cat(graph_text_tokens, dim=0)
    
    # Create input sequences: prompt + graph tokens
    input_ids = torch.cat([prompt_tokens["input_ids"], graph_text_tokens], dim=1)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Create labels for casual language modeling
    labels = input_ids.clone()
    labels[:, :-1] = labels[:, 1:].clone()  # Shift left
    labels[:, -1] = -100  # Ignore last token
    
    # Set prompt tokens to -100 (ignore in loss calculation)
    for i in range(len(batch)):
        prompt_length = prompt_tokens["input_ids"].size(1)
        labels[i, :prompt_length] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "graph_data": [(item["edge_index"], item["x"]) for item in batch],
        "prompt_text": [item["prompt_text"] for item in batch]
    }

# Training function for Stage 1 (Graph Tokenizer)
def train_graph_tokenizer(
    tokenizer,
    train_loader,
    val_loader=None,
    epochs=100,
    learning_rate=0.001,
    weight_decay=1e-5,
    checkpoint_dir="./checkpoints",
    log_wandb=True,
    local_rank=-1,
    ddp=False
):
    """
    Train the graph tokenizer with distributed training
    
    Args:
        tokenizer: Graph tokenizer model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        checkpoint_dir: Directory for saving checkpoints
        log_wandb: Whether to log to wandb
        local_rank: Local rank for distributed training
        ddp: Whether to use Distributed Data Parallel
        
    Returns:
        Trained tokenizer
    """
    # Create optimizer
    optimizer = optim.AdamW(
        tokenizer.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize wandb only on master process
    if log_wandb and (local_rank == 0 or local_rank == -1):
        wandb.init(
            project="hierarchical-graphvq",
            name="graph_tokenizer_training",
            config={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": train_loader.batch_size,
                "K": tokenizer.K,
                "M_list": tokenizer.M_list,
                "distributed": ddp
            }
        )
    
    # Convert model to DDP if needed
    if ddp:
        tokenizer = DDP(tokenizer, device_ids=[local_rank], output_device=local_rank)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        # Set to training mode
        tokenizer.train()
        
        # Reset metrics
        train_loss = 0.0
        recon_loss = 0.0
        vq_loss = 0.0
        commit_loss = 0.0
        batch_count = 0
        
        # Training epoch
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            batch = batch.to(local_rank if ddp else 'cuda')
            edge_index, x = batch.edge_index, batch.x
            
            # Forward pass
            optimizer.zero_grad()
            loss, E, Z, loss_components = tokenizer.module.train_tokenizer(edge_index, x) if ddp else tokenizer.train_tokenizer(edge_index, x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss_components['total_loss']
            recon_loss += loss_components['recon_loss']
            vq_loss += loss_components['vq_loss']
            commit_loss += loss_components['commit_loss']
            batch_count += 1
            
            # Log batch metrics
            if log_wandb and (local_rank == 0 or local_rank == -1) and batch_idx % 10 == 0:
                wandb.log({
                    "batch/total_loss": loss_components['total_loss'],
                    "batch/recon_loss": loss_components['recon_loss'],
                    "batch/vq_loss": loss_components['vq_loss'],
                    "batch/commit_loss": loss_components['commit_loss'],
                    "batch/learning_rate": optimizer.param_groups[0]['lr']
                })
        
        # Calculate average metrics
        train_loss /= batch_count
        recon_loss /= batch_count
        vq_loss /= batch_count
        commit_loss /= batch_count
        
        # Log epoch metrics
        if local_rank == 0 or local_rank == -1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                  f"Recon Loss: {recon_loss:.4f}, VQ Loss: {vq_loss:.4f}, Commit Loss: {commit_loss:.4f}")
            
            if log_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/recon_loss": recon_loss,
                    "train/vq_loss": vq_loss,
                    "train/commit_loss": commit_loss,
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                })
        
        # Validation
        if val_loader is not None:
            tokenizer.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_vq_loss = 0.0
            val_commit_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Move data to device
                    batch = batch.to(local_rank if ddp else 'cuda')
                    edge_index, x = batch.edge_index, batch.x
                    
                    # Forward pass
                    loss, E, Z, loss_components = tokenizer.module.train_tokenizer(edge_index, x) if ddp else tokenizer.train_tokenizer(edge_index, x)
                    
                    # Update metrics
                    val_loss += loss_components['total_loss']
                    val_recon_loss += loss_components['recon_loss']
                    val_vq_loss += loss_components['vq_loss']
                    val_commit_loss += loss_components['commit_loss']
                    val_batch_count += 1
            
            # Calculate average metrics
            val_loss /= val_batch_count
            val_recon_loss /= val_batch_count
            val_vq_loss /= val_batch_count
            val_commit_loss /= val_batch_count
            
            # Log validation metrics
            if local_rank == 0 or local_rank == -1:
                print(f"Validation Loss: {val_loss:.4f}, "
                      f"Recon Loss: {val_recon_loss:.4f}, VQ Loss: {val_vq_loss:.4f}, Commit Loss: {val_commit_loss:.4f}")
                
                if log_wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/recon_loss": val_recon_loss,
                        "val/vq_loss": val_vq_loss,
                        "val/commit_loss": val_commit_loss
                    })
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_loss and (local_rank == 0 or local_rank == -1):
                best_loss = val_loss
                
                # Save checkpoint
                model_to_save = tokenizer.module if ddp else tokenizer
                torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, "best_tokenizer.pt"))
                
                if log_wandb:
                    wandb.run.summary["best_val_loss"] = best_loss
        else:
            # Save checkpoint based on training loss
            if train_loss < best_loss and (local_rank == 0 or local_rank == -1):
                best_loss = train_loss
                
                # Save checkpoint
                model_to_save = tokenizer.module if ddp else tokenizer
                torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, "best_tokenizer.pt"))
                
                if log_wandb:
                    wandb.run.summary["best_train_loss"] = best_loss
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0 and (local_rank == 0 or local_rank == -1):
            model_to_save = tokenizer.module if ddp else tokenizer
            torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, f"tokenizer_epoch_{epoch+1}.pt"))
    
    # Save final model
    if local_rank == 0 or local_rank == -1:
        model_to_save = tokenizer.module if ddp else tokenizer
        torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, "final_tokenizer.pt"))
        
        if log_wandb:
            wandb.finish()
    
    return tokenizer

# Training function for Stage 2 (Graph LLM) with DeepSpeed
def train_graph_llm_deepspeed(
    graph_llm,
    text_tokenizer,
    train_dataset,
    val_dataset=None,
    epochs=50,
    train_batch_size=4,
    gradient_accumulation_steps=1,
    deepspeed_config=None,
    checkpoint_dir="./checkpoints",
    log_wandb=True,
    local_rank=-1,
    world_size=8
):
    """
    Train the graph LLM with DeepSpeed optimization
    
    Args:
        graph_llm: Graph LLM model
        text_tokenizer: Text tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        train_batch_size: Per-GPU batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        deepspeed_config: DeepSpeed configuration
        checkpoint_dir: Directory for saving checkpoints
        log_wandb: Whether to log to wandb
        local_rank: Local rank for distributed training
        world_size: Total number of GPUs
        
    Returns:
        Trained graph LLM
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize DeepSpeed
    if deepspeed_config is None:
        deepspeed_config = create_deepspeed_config(
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
    
    # Create training arguments
    training_args = {
        "fp16": {"enabled": deepspeed_config.get("fp16", {}).get("enabled", False)},
        "zero_optimization": deepspeed_config.get("zero_optimization", {}),
        "train_batch_size": deepspeed_config.get("train_batch_size", train_batch_size * world_size * gradient_accumulation_steps),
        "gradient_accumulation_steps": deepspeed_config.get("gradient_accumulation_steps", gradient_accumulation_steps),
        "gradient_clipping": deepspeed_config.get("gradient_clipping", 1.0),
        "optimizer": deepspeed_config.get("optimizer", None),
        "scheduler": deepspeed_config.get("scheduler", None)
    }
    
    # Initialize wandb only on master process
    if log_wandb and local_rank == 0:
        wandb.init(
            project="hierarchical-graphvq",
            name="graph_llm_training_deepspeed",
            config={
                "epochs": epochs,
                "batch_size": train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "deepspeed_config": deepspeed_config,
                "world_size": world_size
            }
        )
    
    # Create data samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    
    val_sampler = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False
        )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        collate_fn=lambda batch: graph_text_collate_fn(batch, graph_llm),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_batch_size,
            sampler=val_sampler,
            collate_fn=lambda batch: graph_text_collate_fn(batch, graph_llm),
            num_workers=4,
            pin_memory=True
        )
    
    # Initialize DeepSpeed model
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=graph_llm,
        model_parameters=graph_llm.parameters(),
        config=training_args
    )
    
    # Training loop
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(epochs):
        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        
        # Set to training mode
        model_engine.train()
        
        # Reset metrics
        train_loss = 0.0
        batch_count = 0
        
        # Training epoch
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            input_ids = batch["input_ids"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            
            # Forward pass
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            model_engine.backward(loss)
            model_engine.step()
            
            # Update metrics
            train_loss += loss.item()
            batch_count += 1
            global_step += 1
            
            # Log batch metrics
            if log_wandb and local_rank == 0 and batch_idx % 10 == 0:
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/perplexity": torch.exp(loss).item(),
                    "batch/learning_rate": scheduler.get_lr()[0] if scheduler else optimizer.param_groups[0]['lr'],
                    "global_step": global_step
                })
        
        # Calculate average metrics
        train_loss /= batch_count
        
        # Log epoch metrics
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
            
            if log_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/perplexity": torch.exp(torch.tensor(train_loss)).item()
                })
        
        # Validation
        if val_loader is not None:
            model_engine.eval()
            val_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Move data to device
                    input_ids = batch["input_ids"].to(model_engine.device)
                    attention_mask = batch["attention_mask"].to(model_engine.device)
                    labels = batch["labels"].to(model_engine.device)
                    
                    # Forward pass
                    outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_batch_count += 1
            
            # Calculate average metrics
            val_loss /= val_batch_count
            
            # Log validation metrics
            if local_rank == 0:
                print(f"Validation Loss: {val_loss:.4f}")
                
                if log_wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/perplexity": torch.exp(torch.tensor(val_loss)).item()
                    })
            
            # Save best model
            if val_loss < best_loss and local_rank == 0:
                best_loss = val_loss
                
                # Save checkpoint
                model_engine.save_checkpoint(checkpoint_dir, tag="best")
                
                if log_wandb:
                    wandb.run.summary["best_val_loss"] = best_loss
        else:
            # Save checkpoint based on training loss
            if train_loss < best_loss and local_rank == 0:
                best_loss = train_loss
                
                # Save checkpoint
                model_engine.save_checkpoint(checkpoint_dir, tag="best")
                
                if log_wandb:
                    wandb.run.summary["best_train_loss"] = best_loss
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0 and local_rank == 0:
            model_engine.save_checkpoint(checkpoint_dir, tag=f"epoch_{epoch+1}")
        
        # Generate sample graph
        if (epoch + 1) % 10 == 0 and local_rank == 0 and log_wandb:
            try:
                # Set to evaluation mode
                model_engine.eval()
                
                # Generate graph from a sample prompt
                prompt = "Generate a social network with two communities"
                adj_matrix, node_features, Z_list = graph_llm.generate_graph(
                    prompt_text=prompt,
                    text_tokenizer=text_tokenizer,
                    device=model_engine.device
                )
                
                # Visualize graph
                G = nx.from_numpy_array(adj_matrix.cpu().numpy())
                plt.figure(figsize=(8, 8))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, node_size=50, node_color='blue', alpha=0.8)
                plt.title(f"Generated Graph - Epoch {epoch+1}")
                
                # Log to wandb
                wandb.log({
                    "generated_graph": wandb.Image(plt),
                    "prompt": prompt
                })
                plt.close()
            except Exception as e:
                print(f"Error generating sample graph: {e}")
    
    # Save final model
    if local_rank == 0:
        model_engine.save_checkpoint(checkpoint_dir, tag="final")
        
        if log_wandb:
            wandb.finish()
    
    return model_engine

# Main training pipeline
def hierarchical_graphvq_pipeline(args):
    """
    Complete pipeline for training Hierarchical GraphVQ with DeepSpeed
    
    Args:
        args: Command line arguments
    """
    # Set up distributed training
    if args.distributed:
        # Initialize distributed process group
        dist.init_process_group(
            backend='nccl',
            init_method=f'env://'
        )
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        local_rank = -1
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Initialize wandb only on master process
    if args.log_wandb and (local_rank == 0 or local_rank == -1):
        wandb.init(
            project="hierarchical-graphvq",
            name="pipeline",
            config=vars(args)
        )
    
    # Stage 1: Train Graph Tokenizer
    if args.stage1 or args.stage_all:
        if local_rank == 0 or local_rank == -1:
            print("Stage 1: Training Graph Tokenizer")
        
        # Initialize Graph Tokenizer
        tokenizer = GraphTokenizer(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            K=args.K,
            M_list=args.M_list,
            p=args.p,
            num_nodes=args.num_nodes,
            codebook_init=args.codebook_init
        )
        
        # Move tokenizer to device
        tokenizer = tokenizer.to(device)
        
        # Load tokenizer checkpoint if provided
        if args.tokenizer_checkpoint and os.path.exists(args.tokenizer_checkpoint):
            tokenizer.load_state_dict(torch.load(args.tokenizer_checkpoint, map_location=device))
            if local_rank == 0 or local_rank == -1:
                print(f"Loaded tokenizer checkpoint from {args.tokenizer_checkpoint}")
        
        # Create data loaders
        try:
            # Try to load graph dataset
            from torch_geometric.datasets import TUDataset
            
            # Load dataset
            dataset = TUDataset(root=args.data_dir, name=args.dataset_name)
            
            # Split dataset
            train_size = int(len(dataset) * 0.8)
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # Create samplers for distributed training
            if args.distributed:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=local_rank
                )
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=local_rank
                )
            else:
                train_sampler = None
                val_sampler = None
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            # Log dataset info
            if local_rank == 0 or local_rank == -1:
                print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples from {args.dataset_name}")
                
                if args.log_wandb:
                    wandb.log({
                        "dataset_name": args.dataset_name,
                        "train_size": len(train_dataset),
                        "val_size": len(val_dataset)
                    })
            
            # Train tokenizer
            train_graph_tokenizer(
                tokenizer=tokenizer,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.tokenizer_epochs,
                learning_rate=args.tokenizer_lr,
                weight_decay=args.tokenizer_weight_decay,
                checkpoint_dir=os.path.join(args.checkpoint_dir, "tokenizer"),
                log_wandb=args.log_wandb,
                local_rank=local_rank,
                ddp=args.distributed
            )
        except Exception as e:
            if local_rank == 0 or local_rank == -1:
                print(f"Error loading or training with dataset: {e}")
                if args.log_wandb:
                    wandb.log({"error": str(e)})
    
    # Stage 2: Train Graph LLM with DeepSpeed
    if args.stage2 or args.stage_all:
        if local_rank == 0 or local_rank == -1:
            print("Stage 2: Training Graph LLM with DeepSpeed")
        
        # Load pretrained tokenizer
        if args.stage1 and args.stage_all:
            # Use the tokenizer from Stage 1
            graph_tokenizer = tokenizer
        else:
            # Initialize a new tokenizer and load from checkpoint
            graph_tokenizer = GraphTokenizer(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                K=args.K,
                M_list=args.M_list,
                p=args.p,
                num_nodes=args.num_nodes,
                codebook_init=args.codebook_init
            )
            
            # Load tokenizer checkpoint
            tokenizer_checkpoint = args.tokenizer_checkpoint or os.path.join(args.checkpoint_dir, "tokenizer", "best_tokenizer.pt")
            if os.path.exists(tokenizer_checkpoint):
                graph_tokenizer.load_state_dict(torch.load(tokenizer_checkpoint, map_location=device))
                if local_rank == 0 or local_rank == -1:
                    print(f"Loaded tokenizer checkpoint from {tokenizer_checkpoint}")
            else:
                if local_rank == 0 or local_rank == -1:
                    print(f"No tokenizer checkpoint found at {tokenizer_checkpoint}. Using fresh tokenizer.")
        
        # Move tokenizer to device
        graph_tokenizer = graph_tokenizer.to(device)
        
        # Initialize base LLM
        try:
            # Load pretrained model
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
            
            if local_rank == 0 or local_rank == -1:
                print(f"Loading base LLM: {args.base_llm}")
            
            # Load model configuration
            config = AutoConfig.from_pretrained(
                args.base_llm,
                trust_remote_code=True
            )
            
            # Adjust config based on available hardware
            if args.llm_config_override:
                # Apply hardware-specific configuration adjustments
                config.gradient_checkpointing = True  # Enable gradient checkpointing
                if args.hidden_dim:
                    config.hidden_size = args.hidden_dim
                if args.num_layers:
                    config.num_hidden_layers = args.num_layers
                if args.num_heads:
                    config.num_attention_heads = args.num_heads
            
            # Load text tokenizer
            text_tokenizer = AutoTokenizer.from_pretrained(
                args.base_llm,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Ensure pad token exists
            if text_tokenizer.pad_token is None:
                text_tokenizer.pad_token = text_tokenizer.eos_token
            
            # Load base model
            base_llm = AutoModelForCausalLM.from_pretrained(
                args.base_llm,
                config=config,
                trust_remote_code=True
            )
            
            # Initialize GraphLLM
            graph_llm = GraphLLM(
                tokenizer=graph_tokenizer,
                base_llm=base_llm,
                vocab_size=len(text_tokenizer),
                embedding_dim=config.hidden_size,
                hidden_dim=config.hidden_size
            )
            
            # Add special tokens for graph tokens
            text_tokenizer, num_added_tokens = graph_llm.add_special_tokens_to_tokenizer(text_tokenizer)
            if local_rank == 0 or local_rank == -1:
                print(f"Added {num_added_tokens} graph tokens to text tokenizer")
            
            # Load GraphLLM checkpoint if provided
            if args.llm_checkpoint and os.path.exists(args.llm_checkpoint):
                # Load checkpoint
                checkpoint = torch.load(args.llm_checkpoint, map_location=device)
                graph_llm.load_state_dict(checkpoint)
                if local_rank == 0 or local_rank == -1:
                    print(f"Loaded GraphLLM checkpoint from {args.llm_checkpoint}")
            
            # Load or create graph-text dataset
            try:
                # Create synthetic graph-text dataset if needed
                if args.text_data_dir and os.path.exists(args.text_data_dir):
                    # Load text prompts from files
                    import glob
                    
                    text_files = glob.glob(os.path.join(args.text_data_dir, "*.txt"))
                    if local_rank == 0 or local_rank == -1:
                        print(f"Found {len(text_files)} text files in {args.text_data_dir}")
                    
                    # Load text prompts
                    text_prompts = []
                    for file_path in text_files:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text_prompts.append(f.read())
                else:
                    # Create synthetic text prompts
                    text_prompts = [
                        f"Generate a graph with {np.random.randint(3, 10)} nodes and {np.random.randint(2, 15)} edges."
                        for _ in range(100)
                    ]
                    
                    if args.dataset_name:
                        # Add dataset-specific prompts
                        if "ENZYMES" in args.dataset_name:
                            enzyme_prompts = [
                                "Generate an enzyme graph structure.",
                                "Create a protein interaction network.",
                                "Generate a biochemical reaction network.",
                                "Create an enzyme-substrate interaction graph."
                            ]
                            text_prompts.extend([np.random.choice(enzyme_prompts) for _ in range(50)])
                        elif "MUTAG" in args.dataset_name:
                            mutag_prompts = [
                                "Generate a mutagenic compound graph.",
                                "Create a molecular structure with aromatic rings.",
                                "Generate a chemical compound with nitrogen atoms.",
                                "Create a graph of a mutagenic molecule."
                            ]
                            text_prompts.extend([np.random.choice(mutag_prompts) for _ in range(50)])
                
                # Create graph data if needed
                if "train_dataset" not in locals() or "val_dataset" not in locals():
                    # Try to load graph dataset
                    from torch_geometric.datasets import TUDataset
                    
                    # Load dataset
                    dataset = TUDataset(root=args.data_dir, name=args.dataset_name)
                    
                    # Split dataset
                    train_size = int(len(dataset) * 0.8)
                    val_size = len(dataset) - train_size
                    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                
                # Ensure we have matching numbers of graphs and prompts
                n_train = len(train_dataset)
                n_prompts = len(text_prompts)
                
                if n_prompts < n_train:
                    # Repeat prompts if needed
                    text_prompts = text_prompts * (n_train // n_prompts + 1)
                
                # Truncate prompts to match number of graphs
                text_prompts = text_prompts[:n_train]
                
                # Create graph-text dataset
                train_graph_text_dataset = GraphTextDataset(
                    text_tokenizer=text_tokenizer,
                    graph_tokenizer=graph_tokenizer,
                    graph_data=[(sample.edge_index, sample.x) for sample in train_dataset],
                    text_prompts=text_prompts[:n_train],
                    max_length=args.max_length
                )
                
                # Create validation dataset
                val_text_prompts = [
                    f"Generate a graph with {np.random.randint(3, 10)} nodes and {np.random.randint(2, 15)} edges."
                    for _ in range(len(val_dataset))
                ]
                
                val_graph_text_dataset = GraphTextDataset(
                    text_tokenizer=text_tokenizer,
                    graph_tokenizer=graph_tokenizer,
                    graph_data=[(sample.edge_index, sample.x) for sample in val_dataset],
                    text_prompts=val_text_prompts,
                    max_length=args.max_length
                )
                
                if local_rank == 0 or local_rank == -1:
                    print(f"Created graph-text datasets with {len(train_graph_text_dataset)} training and {len(val_graph_text_dataset)} validation samples")
                
                # Create DeepSpeed configuration
                if args.deepspeed_config and os.path.exists(args.deepspeed_config):
                    # Load DeepSpeed config from file
                    with open(args.deepspeed_config, 'r') as f:
                        deepspeed_config = json.load(f)
                else:
                    # Create default DeepSpeed config
                    deepspeed_config = create_deepspeed_config(
                        stage=2,  # ZeRO stage 2
                        offload_optimizer=args.offload_optimizer,
                        offload_parameters=args.offload_parameters,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        fp16=args.fp16,
                        learning_rate=args.llm_lr,
                        warmup_steps=args.warmup_steps,
                        gradient_clipping=args.gradient_clipping,
                        train_batch_size=args.llm_batch_size
                    )
                
                # Train GraphLLM with DeepSpeed
                graph_llm_engine = train_graph_llm_deepspeed(
                    graph_llm=graph_llm,
                    text_tokenizer=text_tokenizer,
                    train_dataset=train_graph_text_dataset,
                    val_dataset=val_graph_text_dataset,
                    epochs=args.llm_epochs,
                    train_batch_size=args.llm_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    deepspeed_config=deepspeed_config,
                    checkpoint_dir=os.path.join(args.checkpoint_dir, "llm"),
                    log_wandb=args.log_wandb,
                    local_rank=local_rank,
                    world_size=world_size
                )
                
                if local_rank == 0 or local_rank == -1:
                    print("Stage 2: GraphLLM training completed")
            
            except Exception as e:
                if local_rank == 0 or local_rank == -1:
                    print(f"Error during GraphLLM training: {e}")
                    import traceback
                    traceback.print_exc()
                    if args.log_wandb:
                        wandb.log({"error": str(e)})
        
        except Exception as e:
            if local_rank == 0 or local_rank == -1:
                print(f"Error initializing base LLM: {e}")
                import traceback
                traceback.print_exc()
                if args.log_wandb:
                    wandb.log({"error": str(e)})
    
    # Clean up
    if args.distributed:
        dist.destroy_process_group()
    
    if args.log_wandb and (local_rank == 0 or local_rank == -1):
        wandb.finish()

# Main function
def main():
    """Main function to parse arguments and run the pipeline"""
    parser = argparse.ArgumentParser(description="Hierarchical GraphVQ with DeepSpeed")
    
    # Pipeline options
    parser.add_argument("--stage1", action="store_true", help="Run Stage 1 (Graph Tokenizer) training")
    parser.add_argument("--stage2", action="store_true", help="Run Stage 2 (Graph LLM) training")
    parser.add_argument("--stage_all", action="store_true", help="Run both stages")
    
    # Distributed training options
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # General options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="hierarchical-graphvq", help="Weights & Biases project name")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory for saving checkpoints")
    
    # Data options
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--dataset_name", type=str, default="ENZYMES", help="Dataset name")
    parser.add_argument("--text_data_dir", type=str, default=None, help="Text data directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Model options
    parser.add_argument("--input_dim", type=int, default=3, help="Input dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--K", type=int, default=3, help="Number of hops")
    parser.add_argument("--M_list", type=int, nargs="+", default=[256, 128, 64], help="Codebook sizes for each hop")
    parser.add_argument("--p", type=int, default=20, help="Maximum number of neighbors to sample")
    parser.add_argument("--num_nodes", type=int, default=100, help="Number of nodes in the graph")
    parser.add_argument("--codebook_init", type=str, default="kmeans", choices=["random", "kmeans"], help="Codebook initialization method")
    parser.add_argument("--base_llm", type=str, default="gpt2", help="Base LLM model name")
    parser.add_argument("--llm_config_override", action="store_true", help="Override LLM configuration for hardware")
    parser.add_argument("--num_layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    # Checkpoint options
    parser.add_argument("--tokenizer_checkpoint", type=str, default=None, help="Path to tokenizer checkpoint")
    parser.add_argument("--llm_checkpoint", type=str, default=None, help="Path to LLM checkpoint")
    
    # Training options (Stage 1)
    parser.add_argument("--tokenizer_epochs", type=int, default=100, help="Number of tokenizer training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for tokenizer training")
    parser.add_argument("--tokenizer_lr", type=float, default=0.001, help="Learning rate for tokenizer training")
    parser.add_argument("--tokenizer_weight_decay", type=float, default=1e-5, help="Weight decay for tokenizer training")
    
    # Training options (Stage 2)
    parser.add_argument("--llm_epochs", type=int, default=50, help="Number of LLM training epochs")
    parser.add_argument("--llm_batch_size", type=int, default=4, help="Batch size for LLM training (per GPU)")
    parser.add_argument("--llm_lr", type=float, default=5e-5, help="Learning rate for LLM training")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="Gradient clipping threshold")
    
    # DeepSpeed options
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed configuration file")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--offload_optimizer", action="store_true", help="Offload optimizer states to CPU")
    parser.add_argument("--offload_parameters", action="store_true", help="Offload parameters to CPU")
    
    args = parser.parse_args()
    
    # Set local_rank from environment if distributed
    if args.distributed and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    
    # Run pipeline
    hierarchical_graphvq_pipeline(args)

if __name__ == "__main__":
    main(): Train Graph LLM with DeepSpeed
    if args.stage2 or args.stage_all:
        if local_rank == 0 or local_rank == -1:
            print("Stage 2import os
import argparse
import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader
import wandb
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# GNN Encoder
class GNNEncoder(nn.Module):
    """
    GNN encoder for k-hop node representation calculation
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, e_prev, edge_index, batch=None):
        """
        Compute node representations for the current hop
        """
        x = self.conv1(e_prev, edge_index)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        return x
    
    def sample_neighbors(self, edge_index, num_nodes, max_size=20):
        """
        Sample neighbors for each node with maximum size p
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
        
        if len(sampled_edges) == 0:
            # If no edges, return empty tensor with correct shape
            return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
        
        return torch.tensor(sampled_edges).t().contiguous().to(edge_index.device)

# Graph Decoder
class GraphDecoder(nn.Module):
    """
    Graph decoder to reconstruct graph structure from codebook embeddings
    """
    def __init__(self, embedding_dim, hidden_dim, num_nodes, dropout=0.1):
        super(GraphDecoder, self).__init__()
        self.num_nodes = num_nodes
        
        # More powerful node predictor with residual connections
        self.node_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Edge predictor with MLP
        self.edge_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_embeddings):
        """
        Reconstruct graph structure from node embeddings
        """
        # Reconstruct node features
        node_features = self.node_predictor(node_embeddings)
        
        # Reconstruct adjacency matrix more efficiently
        # Create all pairs of node embeddings
        n = node_embeddings.size(0)
        
        # Create indices for all pairs
        idx_i = torch.arange(n, device=node_embeddings.device).repeat_interleave(n)
        idx_j = torch.arange(n, device=node_embeddings.device).repeat(n)
        
        # Skip self-loops by creating a mask
        mask = (idx_i != idx_j)
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        
        # Get embeddings for each pair
        emb_i = node_embeddings[idx_i]
        emb_j = node_embeddings[idx_j]
        
        # Concatenate and predict edges
        edge_features = torch.cat([emb_i, emb_j], dim=1)
        edge_preds = self.edge_predictor(edge_features).squeeze()
        
        # Create adjacency matrix from predictions
        adj_matrix = torch.zeros(n, n, device=node_embeddings.device)
        adj_matrix[idx_i, idx_j] = edge_preds
                    
        return adj_matrix, node_features

# Graph Tokenizer
class GraphTokenizer(nn.Module):
    """
    Hierarchical Graph Vector Quantizer (GraphVQ) for tokenizing graphs
    """
    def __init__(self, input_dim, hidden_dim, K, M_list, p, num_nodes, codebook_init='random'):
        """
        Initialize the Graph Tokenizer
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            K: Maximum number of hops
            M_list: List of codebook sizes for each hop [M^0, M^1, ..., M^K]
            p: Maximum number of neighbors to sample
            num_nodes: Number of nodes in the graph
            codebook_init: Initialization method for codebooks ('random' or 'kmeans')
        """
        super(GraphTokenizer, self).__init__()
        self.K = K
        self.M_list = M_list
        self.p = p
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.codebook_init = codebook_init
        self.codebook_temp = 1.0  # Temperature for codebook assignment, can be annealed
        
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
        
        # Initialize codebook usage tracking (useful for monitoring)
        self.register_buffer('codebook_usage', torch.zeros(K, max(M_list)))
        
        # Exponential moving average decay for codebook updates
        self.ema_decay = 0.99
        
        # Initialize decoder with dropout
        self.decoder = GraphDecoder(hidden_dim, hidden_dim, num_nodes, dropout=0.1)
        
        # Storage for last assignments (useful for logging)
        self.last_assignments = []
    
    def build_codebook(self, embeddings, M, k, init_if_needed=True):
        """
        Build or update codebook for the k-th hop
        
        Args:
            embeddings: Node embeddings for current hop
            M: Codebook size for current hop
            k: Current hop index
            init_if_needed: Whether to initialize codebook if not already initialized
            
        Returns:
            Z_k: Codebook indices
            E_k: Codebook embeddings
        """
        # Check if codebook needs to be initialized
        codebook_initialized = torch.any(self.codebooks[k, :M] != 0)
        
        if not codebook_initialized and init_if_needed:
            # Initialize codebook
            if self.codebook_init == 'kmeans' and embeddings.size(0) > M:
                # Use K-means clustering to create codebook
                embeddings_np = embeddings.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=M, random_state=0, n_init=10).fit(embeddings_np)
                
                # Get cluster centers as codebook embeddings
                E_k = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(embeddings.device)
                
                # Get cluster assignments as codebook indices
                Z_k = torch.tensor(kmeans.labels_, dtype=torch.long).to(embeddings.device)
            else:
                # Randomly initialize codebook from embeddings
                perm = torch.randperm(embeddings.size(0))
                if M <= embeddings.size(0):
                    # If we have enough embeddings, sample from them
                    idx = perm[:M]
                    E_k = embeddings[idx].clone()
                else:
                    # If not enough embeddings, repeat some and add noise
                    idx = perm[:(M % embeddings.size(0))]
                    repeats = M // embeddings.size(0)
                    E_k = torch.cat([embeddings.clone()] * repeats + [embeddings[idx].clone()], dim=0)
                    # Add small noise to avoid duplicates
                    E_k += torch.randn_like(E_k) * 0.01
                
                # Assign each embedding to nearest codebook entry
                Z_k = self.get_nearest_indices(embeddings, E_k)
            
            # Update codebook
            self.codebooks[k, :M] = E_k
        else:
            # Use existing codebook
            E_k = self.codebooks[k, :M]
            # Assign each embedding to nearest codebook entry
            Z_k = self.get_nearest_indices(embeddings, E_k)
        
        return Z_k, E_k
    
    def get_nearest_indices(self, embeddings, codebook):
        """Get indices of nearest codebook entries for each embedding"""
        # Calculate distances between embeddings and codebook entries
        distances = torch.cdist(embeddings, codebook)
        
        # Find nearest codebook entry for each embedding
        indices = torch.argmin(distances, dim=1)
        
        return indices
    
    def update_codebook_ema(self, embeddings, indices, k, M):
        """Update codebook with exponential moving average"""
        if self.training:
            # One-hot encoding of indices
            one_hot = F.one_hot(indices, num_classes=M).float()
            
            # Calculate new codebook values with EMA
            # Sum of embeddings assigned to each codebook entry
            sum_embeddings = torch.matmul(one_hot.t(), embeddings)
            # Sum of assignments to each codebook entry
            sum_assignments = one_hot.sum(dim=0).unsqueeze(1)
            
            # Avoid division by zero
            sum_assignments = torch.max(sum_assignments, torch.ones_like(sum_assignments))
            
            # Update codebook with EMA
            new_codebook = sum_embeddings / sum_assignments
            self.codebooks[k, :M] = self.ema_decay * self.codebooks[k, :M] + (1 - self.ema_decay) * new_codebook
            
            # Update codebook usage for monitoring
            self.codebook_usage[k, :M] = self.ema_decay * self.codebook_usage[k, :M] + (1 - self.ema_decay) * one_hot.sum(dim=0)
    
    def assign_nearest(self, embeddings, codebook):
        """
        Assign each embedding to the nearest codebook entry
        
        Args:
            embeddings: Node embeddings
            codebook: Codebook embeddings
            
        Returns:
            Quantized embeddings and indices
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
        Train the graph tokenizer
        
        Args:
            edge_index: Graph connectivity in COO format
            X: Node features
            
        Returns:
            Loss value, list of embeddings, and list of codebook indices
        """
        # Initialize: E^0 = X
        E = [X]
        Z = []
        
        # Store assignments for loss calculation
        assignments = []
        
        for k in range(self.K):
            # Step 1: Calculate k-hop node representations
            h_k = self.gnn_encoders[k](
                e_prev=E[-1], 
                edge_index=self.gnn_encoders[k].sample_neighbors(
                    edge_index=edge_index, 
                    num_nodes=self.num_nodes, 
                    max_size=self.p
                )
            )
            
            # Step 2: Build codebook and embeddings
            Z_k, E_k = self.build_codebook(h_k, M=self.M_list[k], k=k)
            Z.append(Z_k)
            
            # Step 3: Assign nearest embeddings
            quantized, indices = self.assign_nearest(h_k, E_k)
            E.append(quantized)
            
            # Step 4: Update codebook with EMA (if training)
            if self.training:
                self.update_codebook_ema(h_k, indices, k, self.M_list[k])
            
            # Store for loss calculation
            assignments.append((h_k, quantized, indices))
        
        # Store assignments for external access (for wandb logging)
        self.last_assignments = assignments
        
        # Calculate loss
        loss, loss_components = self.calculate_loss(edge_index, X, E, assignments)
        
        return loss, E, Z, loss_components
    
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
                if i < len(Z_list[k]):  # Check if index is valid
                    codebook_idx = Z_list[k][i].item()
                    if codebook_idx < self.M_list[k]:  # Check if codebook index is valid
                        final_embeddings[i] = self.codebooks[k, codebook_idx]
            
            # Decode graph from embeddings
            adj_matrix, node_features = self.decoder(final_embeddings)
            
            return adj_matrix, node_features

# Graph LLM
class GraphLLM(nn.Module):
    """
    Language model for graph generation with DeepSpeed optimization
    """
    def __init__(self, tokenizer, base_llm, vocab_size, embedding_dim, hidden_dim):
        """
        Initialize the Graph LLM
        
        Args:
            tokenizer: Pretrained Graph Tokenizer
            base_llm: Base language model
            vocab_size: Original vocabulary size
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden layers
        """
        super(GraphLLM, self).__init__()
        self.tokenizer = tokenizer
        self.base_llm = base_llm
        
        # Calculate number of special tokens for graph
        self.num_graph_tokens = sum(tokenizer.M_list) + 2 * (tokenizer.K + 1)  # Include start/end tokens
        
        # Mapping from graph tokens to text token IDs
        self.special_tokens = {}
        self.graph_token_map = {}
        
        # DeepSpeed specific optimizations: make model compatible with DeepSpeed
        self.fp16_enabled = False  # Will be set by DeepSpeed
        
    def add_special_tokens_to_tokenizer(self, text_tokenizer):
        """
        Add special tokens to the text tokenizer and set up token maps
        
        Args:
            text_tokenizer: Text tokenizer to extend
            
        Returns:
            Extended tokenizer and number of added tokens
        """
        special_tokens = []
        
        # Add structural tokens
        special_tokens.append("<graph>")  # START_GRAPH
        special_tokens.append("</graph>")  # END_GRAPH
        
        # Add hop tokens
        for k in range(self.tokenizer.K):
            special_tokens.append(f"<hop_{k}>")  # START_HOP_k
            special_tokens.append(f"</hop_{k}>")  # END_HOP_k
        
        # Add codebook tokens
        for k in range(self.tokenizer.K):
            for m in range(self.tokenizer.M_list[k]):
                special_tokens.append(f"z_{m}^{k}")
        
        # Add special tokens to the tokenizer
        tokens_to_add = {"additional_special_tokens": special_tokens}
        num_added_tokens = text_tokenizer.add_special_tokens(tokens_to_add)
        
        # Ensure pad token exists
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token
        
        # Map special structure tokens
        self.special_tokens = {
            'START_GRAPH': text_tokenizer.convert_tokens_to_ids("<graph>"),
            'END_GRAPH': text_tokenizer.convert_tokens_to_ids("</graph>")
        }
        
        # Map hop tokens
        for k in range(self.tokenizer.K):
            self.special_tokens[f'START_HOP_{k}'] = text_tokenizer.convert_tokens_to_ids(f"<hop_{k}>")
            self.special_tokens[f'END_HOP_{k}'] = text_tokenizer.convert_tokens_to_ids(f"</hop_{k}>")
        
        # Map codebook tokens
        for k in range(self.tokenizer.K):
            for m in range(self.tokenizer.M_list[k]):
                token_name = f"z_{m}^{k}"
                self.graph_token_map[(k, m)] = text_tokenizer.convert_tokens_to_ids(token_name)
        
        # Resize token embeddings in base model
        self.base_llm.resize_token_embeddings(len(text_tokenizer))
        
        return text_tokenizer, num_added_tokens
    
    def graph_to_tokens(self, Z_list):
        """
        Convert graph tokens to text token IDs
        
        Args:
            Z_list: List of token indices for each hop
            
        Returns:
            text_tokens: Token IDs for text model
        """
        text_tokens = []
        
        # Add start token for graph
        text_tokens.append(self.special_tokens['START_GRAPH'])
        
        # Add tokens for each hop
        for k in range(self.tokenizer.K):
            # Add start token for hop
            text_tokens.append(self.special_tokens[f'START_HOP_{k}'])
            
            # Add tokens for each node
            for i in range(len(Z_list[k])):
                m = Z_list[k][i].item()
                token_id = self.graph_token_map.get((k, m), 0)
                text_tokens.append(token_id)
            
            # Add end token for hop
            text_tokens.append(self.special_tokens[f'END_HOP_{k}'])
        
        # Add end token for graph
        text_tokens.append(self.special_tokens['END_GRAPH'])
        
        return torch.tensor([text_tokens], device=Z_list[0].device)
    
    def tokens_to_graph(self, tokens):
        """
        Convert text token IDs to graph tokens
        
        Args:
            tokens: Token IDs from text model
            
        Returns:
            Z_list: List of token indices for each hop
        """
        Z_list = [[] for _ in range(self.tokenizer.K)]
        current_hop = -1
        parsing_hop = False
        
        # Parse tokens
        for token in tokens:
            token = token.item()
            
            # Check if it's a graph start token
            if token == self.special_tokens.get('START_GRAPH', -1):
                parsing_hop = True
                continue
                
            # Check if it's a graph end token
            if token == self.special_tokens.get('END_GRAPH', -1):
                parsing_hop = False
                break
                
            if not parsing_hop:
                continue
                
            # Check if it's a hop start token
            start_hop = False
            for k in range(self.tokenizer.K):
                if token == self.special_tokens.get(f'START_HOP_{k}', -1):
                    current_hop = k
                    start_hop = True
                    break
                    
            if start_hop:
                continue
                
            # Check if it's a hop end token
            end_hop = False
            for k in range(self.tokenizer.K):
                if token == self.special_tokens.get(f'END_HOP_{k}', -1):
                    current_hop = -1
                    end_hop = True
                    break
                    
            if end_hop:
                continue
                
            # If we're in a hop section, check if it's a graph token
            if current_hop >= 0:
                # Find which codebook entry this token represents
                for m in range(self.tokenizer.M_list[current_hop]):
                    if token == self.graph_token_map.get((current_hop, m), -1):
                        Z_list[current_hop].append(m)
                        break
        
        # Convert to tensors
        for k in range(self.tokenizer.K):
            if Z_list[k]:  # Only convert non-empty lists
                Z_list[k] = torch.tensor(Z_list[k], device=tokens.device)
            else:
                Z_list[k] = torch.tensor([], device=tokens.device, dtype=torch.long)
        
        return Z_list
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for next token prediction
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for language modeling
            
        Returns:
            outputs: Model outputs including loss and logits
        """
        # Forward pass through base LLM
        if labels is not None:
            outputs = self.base_llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            outputs = self.base_llm(input_ids=input_ids, attention_mask=attention_mask)
        
        return outputs
    
    def generate_graph(self, prompt_text, text_tokenizer, max_length=512, num_beams=5, device='cuda'):
        """
        Generate graph from text prompt
        
        Args:
            prompt_text: Input text prompt
            text_tokenizer: Text tokenizer
            max_length: Maximum sequence length
            num_beams: Number of beams for beam search
            device: Device to use for generation
            
        Returns:
            adj_matrix: Generated adjacency matrix
            node_features: Generated node features
            Z_list: List of token indices for each hop
        """
        # Tokenize prompt
        encoded_prompt = text_tokenizer(prompt_text, return_tensors="pt").to(device)
        input_ids = encoded_prompt.input_ids
        
        # Generate graph tokens
        with torch.no_grad():
            # Generate with beam search
            generation_kwargs = {
                'input_ids': input_ids,
                'max_length': max_length,
                'num_beams': num_beams,
                'do_sample': True,
                'top_p': 0.95,
                'temperature': 0.8,
                'pad_token_id': text_tokenizer.pad_token_id,
                'eos_token_id': self.special_tokens['END_GRAPH'],
                'no_repeat_ngram_size': 3,
                'early_stopping': True
            }
            
            # Generate output tokens
            output_ids = self.base_llm.generate(**generation_kwargs)
        
        # Extract graph tokens (skip the prompt tokens)
        graph_tokens = output_ids[0, input_ids.size(1):]
        
        # Convert to graph tokens
        Z_list = self.tokens_to_graph(graph_tokens)
        
        # Decode graph from tokens
        adj_matrix, node_features = self.tokenizer.decode_graph(Z_list)
        
        return adj_matrix, node_features, Z_list

# Create DeepSpeed config
def create_deepspeed_config(
    stage=2,  # ZeRO optimization stage (0, 1, 2, 3)
    offload_optimizer=False,  # Offload optimizer to CPU
    offload_parameters=False,  # Offload parameters to CPU
    gradient_accumulation_steps=1,
    fp16=True,  # Use fp16 for training
    learning_rate=5e-5,
    warmup_steps=100,
    gradient_clipping=1.0,
    train_batch_size=4,  # Per-GPU batch size
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8
):
    """
    Create DeepSpeed configuration
    
    Args:
        stage: ZeRO optimization stage
        offload_optimizer: Whether to offload optimizer states to CPU
        offload_parameters: Whether to offload parameters to CPU
        gradient_accumulation_steps: Number of gradient accumulation steps
        fp16: Whether to use fp16 for training
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        gradient_clipping: Gradient clipping threshold
        train_batch_size: Per-GPU batch size