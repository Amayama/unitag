import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup


class GraphProjector(nn.Module):
    """
    图嵌入投影模块，支持线性投影和双层MLP投影
    """
    def __init__(self, 
                 embedding_dim, 
                 lm_hidden_dim, 
                 projection_type="linear",
                 projection_dim=None,
                 dropout_rate=0.1):
        super().__init__()
        
        if projection_dim is None:
            projection_dim = int((embedding_dim * lm_hidden_dim) ** 0.5)
        
        if projection_type == "linear":
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, lm_hidden_dim),
                nn.LayerNorm(lm_hidden_dim)
            )
        elif projection_type == "2-layer-mlp":
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(projection_dim, lm_hidden_dim)
            )
        else:
            raise ValueError(f"不支持的投影类型: {projection_type}")
    
    def forward(self, x):
        return self.projection(x)
