import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
import numpy as np
import random
from tqdm import tqdm
import wandb
import os
import argparse
from torch_geometric.utils import to_networkx
import networkx as nx

class GraphEmbeddingDataset(Dataset):
    """图节点嵌入及其对应文本描述的数据集"""
    
    def __init__(self, embeddings, texts, tokenizer, max_length=4096):
        if isinstance(embeddings, dict):
            self.embeddings = list(embeddings.values())
            self.texts = [texts[k] for k in embeddings.keys()]
        else:
            self.embeddings = embeddings
            self.texts = texts
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.bfloat16)
        
        # 编码文本以进行下一个标记预测
        encoded = self.tokenizer(
            self.texts[idx], 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)
        
        # 对于因果LM，标签与input_ids相同（下一个标记预测）
        labels = input_ids.clone()
        
        return {
            "embedding": embedding,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
