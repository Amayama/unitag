class GraphLLM(nn.Module):
    """
    Language model for graph generation
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
        self.num_graph_tokens = sum(self.tokenizer.M_list)
        
        # Extend token embeddings for graph tokens
        self.token_embedding = nn.Embedding(vocab_size + self.num_graph_tokens, embedding_dim)
        
        # Copy original embeddings if base_llm has them
        if hasattr(base_llm, 'token_embedding'):
            self.token_embedding.weight.data[:vocab_size] = base_llm.token_embedding.weight.data
        
        # Initialize new token embeddings for graph tokens
        nn.init.normal_(self.token_embedding.weight.data[vocab_size:], mean=0.0, std=0.02)
        
        # Mapping from graph tokens to text token IDs
        self.graph_token_map = {}
        current_idx = vocab_size
        
        for k in range(self.tokenizer.K):
            for m in range(self.tokenizer.M_list[k]):
                token_name = f"z_{m}^{k}"
                self.graph_token_map[(k, m)] = current_idx
                current_idx += 1
    
    def add_special_tokens(self, tokenizer):
        """
        Add special tokens to the text tokenizer
        
        Args:
            tokenizer: Text tokenizer
        """
        special_tokens = []
        
        for k in range(self.tokenizer.K):
            for m in range(self.tokenizer.M_list[k]):
                special_tokens.append(f"z_{m}^{k}")
                
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        return tokenizer
    
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
        text_tokens.append(self.graph_token_map.get('START_GRAPH', 0))
        
        # Add tokens for each hop
        for k in range(self.tokenizer.K):
            # Add start token for hop
            text_tokens.append(self.graph_token_map.get(f'START_HOP_{k}', 0))
            
            # Add tokens for each node
            for i in range(len(Z_list[k])):
                m = Z_list[k][i].item()
                text_tokens.append(self.graph_token_map.get((k, m), 0))
            
            # Add end token for hop
            text_tokens.append(self.graph_token_map.get(f'END_HOP_{k}', 0))
        
        # Add end token for graph
        text_tokens.append(self.graph_token_map.get('END_GRAPH', 0))
        
        return torch.tensor(text_tokens)
    
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
        
        # Parse tokens
        for token in tokens:
            token = token.item()
            
            # Check if it's a hop start token
            if token in [self.graph_token_map.get(f'START_HOP_{k}', -1) for k in range(self.tokenizer.K)]:
                for k in range(self.tokenizer.K):
                    if token == self.graph_token_map.get(f'START_HOP_{k}', -1):
                        current_hop = k
                        break
                continue
            
            # Check if it's a hop end token
            if token in [self.graph_token_map.get(f'END_HOP_{k}', -1) for k in range(self.tokenizer.K)]:
                current_hop = -1
                continue
            
            # Check if it's a graph token
            for m in range(self.tokenizer.M_list[current_hop]):
                if token == self.graph_token_map.get((current_hop, m), -1):
                    Z_list[current_hop].append(m)
                    break
        
        # Convert to tensors
        for k in range(self.tokenizer.K):
            Z_list[k] = torch.tensor(Z_list[k])
        
        return Z_list
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for next token prediction
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            logits: Next token prediction logits
        """
        # Get embeddings from extended vocabulary
        embeddings = self.token_embedding(input_ids)
        
        # Forward pass through base LLM
        outputs = self.base_llm(inputs_embeds=embeddings, attention_mask=attention_mask)
        
        return outputs
    
    def generate_graph(self, prompt_text, tokenizer, max_length=100):
        """
        Generate graph from text prompt
        
        Args:
            prompt_text: Input text prompt
            tokenizer: Text tokenizer
            max_length: Maximum sequence length
            
        Returns:
            adj_matrix: Generated adjacency matrix
            node_features: Generated node features
        """
        # Tokenize prompt
        prompt_tokens = tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
        
        # Generate tokens
        output_tokens = self.base_llm.generate(
            input_ids=prompt_tokens,
            max_length=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        
        # Extract graph tokens
        graph_tokens = output_tokens[0, prompt_tokens.shape[1]:]
        
        # Convert to graph tokens
        Z_list = self.tokens_to_graph(graph_tokens)
        
        # Decode graph from tokens
        adj_matrix, node_features = self.tokenizer.decode_graph(Z_list)
        
        return adj_matrix, node_features