from torch.utils.data import Dataset

class TextGraphDataset(Dataset):
                    def __init__(self, graph_data, text_tokenizer, num_samples=100, max_length=64):
                        self.graph_data = graph_data
                        self.text_tokenizer = text_tokenizer
                        self.num_samples = num_samples
                        self.max_length = max_length
                        
                        # Create prompts based on dataset type
                        if args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
                            # For citation networks, create paper-related prompts
                            self.prompts = [
                                f"Generate a citation network similar to {args.dataset}",
                                f"Create a graph of {args.dataset} paper citations",
                                f"Produce a citation graph in the style of {args.dataset}",
                                f"Generate a network of papers like {args.dataset}",
                                f"Show me a {args.dataset}-like citation structure"
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
                        
                        # Get graph
                        if args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
                            graph = self.graph_data[idx % len(self.graph_data)]
                        else:
                            graph = self.graph_data[idx % len(self.graph_data)]
                        
                        return prompt_tokens, graph