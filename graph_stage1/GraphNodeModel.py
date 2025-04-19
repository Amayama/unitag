import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
import numpy as np
import random
from GraphProjector import *
import wandb
import os
import argparse
from torch_geometric.utils import to_networkx
import networkx as nx
class GraphNodeModel(nn.Module):
    """
    基于LLM的图节点解释模型
    """
    def __init__(self, 
                 embedding_dim,
                 lm_model_name="Vicuna-7b", 
                 projection_type="linear",
                 projection_dim=None,
                 freeze_backbone=True,
                 prompt_length=5,
                 bf16=True):
        super().__init__()
        
        # 加载语言模型
        # 显式指定设备映射，合理分布到多个GPU
        self.lm = AutoModelForCausalLM.from_pretrained(
            lm_model_name, 
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            device_map="auto",
            # 添加低资源设置
            low_cpu_mem_usage=True,
            # 添加量化配置以减少内存使用
            quantization_config=None  # 如果需要，可以指定量化配置
        )
        lm_hidden_dim = self.lm.config.hidden_size
        
        # 创建投影模块
        self.projector = GraphProjector(
            embedding_dim=embedding_dim,
            lm_hidden_dim=lm_hidden_dim,
            projection_type=projection_type,
            projection_dim=projection_dim
        )
        
        # 冻结语言模型参数
        if freeze_backbone:
            for param in self.lm.parameters():
                param.requires_grad = False
        
        # 提示词嵌入（可学习）
        self.prompt_length = prompt_length
        self.prompt_embeddings = nn.Parameter(
            torch.randn(self.prompt_length, lm_hidden_dim) * 0.02
        )
        
        # 特殊标记
        self.graph_token_id = -200  # 图标记占位符
        self.graph_pad_token_id = -500  # 图填充标记
        
    def forward(self, graph_embeddings, input_ids=None, attention_mask=None, labels=None):
        batch_size = graph_embeddings.size(0)
        
        # 投影图嵌入到LM空间
        projected_embeddings = self.projector(graph_embeddings)
        
        # 获取LM的输入嵌入
        lm_inputs = self.lm.get_input_embeddings()(input_ids)
        
        # 为批次中的每个项添加投影嵌入和提示词嵌入
        graph_embed_expanded = projected_embeddings.unsqueeze(1)
        prompt_embeds_expanded = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接 [图嵌入, 提示词嵌入, 语言模型输入]
        augmented_embeds = torch.cat([
            graph_embed_expanded,
            prompt_embeds_expanded,
            lm_inputs
        ], dim=1)
        
        # 创建新的注意力掩码以考虑添加的嵌入
        new_attn_size = attention_mask.size(1) + 1 + self.prompt_length
        new_attention_mask = torch.ones(batch_size, new_attn_size, device=attention_mask.device)
        
        # 调整标签以考虑添加的嵌入（在损失计算中忽略它们）
        new_labels = torch.full((batch_size, new_attn_size), -100, dtype=torch.long, device=labels.device)
        new_labels[:, 1 + self.prompt_length:] = labels
        
        # 启用梯度检查点以优化内存使用
        self.lm.gradient_checkpointing_enable()
        
        # 使用自定义嵌入通过LM进行前向传递
        # 添加内存优化选项
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            outputs = self.lm(
                inputs_embeds=augmented_embeds,
                attention_mask=new_attention_mask,
                labels=new_labels,
                return_dict=True,
                # 添加以下参数来减少内存使用
                output_attentions=False,  # 不输出注意力权重
                output_hidden_states=False,  # 不输出隐藏状态
                use_cache=False  # 不使用过去的键值对缓存
            )
        
        return outputs