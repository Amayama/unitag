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
import deepspeed

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
        self.lm = AutoModelForCausalLM.from_pretrained(
            lm_model_name, 
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            device_map="auto"
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
        outputs = self.lm(
            inputs_embeds=augmented_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            return_dict=True
        )
        
        return outputs
    
    def generate_description(self, graph_embedding, tokenizer, max_length=50):
        """为图节点生成文本描述"""
        with torch.no_grad():
            # 投影图嵌入到LM隐藏维度
            projected_embedding = self.projector(graph_embedding.unsqueeze(0))
            
            # 准备提示词嵌入
            graph_embed_expanded = projected_embedding.unsqueeze(1)
            prompt_embeds_expanded = self.prompt_embeddings.unsqueeze(0)
            
            # 初始输入仅包含BOS标记
            input_ids = torch.tensor([[tokenizer.bos_token_id]], device=graph_embedding.device)
            input_embeds = self.lm.get_input_embeddings()(input_ids)
            
            # 拼接初始输入的嵌入
            inputs_embeds = torch.cat([
                graph_embed_expanded,
                prompt_embeds_expanded,
                input_embeds
            ], dim=1)
            
            # 使用嵌入生成文本
            outputs = self.lm.generate(
                inputs_embeds=inputs_embeds,
                max_length=max_length + 1 + self.prompt_length + 1,
                num_beams=5,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=False,
                return_dict_in_generate=False
            )
            
            # 删除与嵌入和提示对应的前几个标记
            output_ids = outputs[0, 1+self.prompt_length:]
            
            # 解码为文本
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            return generated_text


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


def process_graph_structure(graph, node_idx, use_hop=2, sample_size=10):
    """
    处理图结构以获取节点的多跳邻居信息
    
    Args:
        graph: NetworkX图
        node_idx: 中心节点索引
        use_hop: 跳数
        sample_size: 每个节点采样的邻居数
        
    Returns:
        结构嵌入
    """
    # 初始化结构嵌入，大小为(sample_size^(use_hop+1)-1)/(sample_size-1)
    structure_dim = int((sample_size**(use_hop+1)-1)/(sample_size-1))
    structure_embedding = np.zeros(structure_dim)
    
    # BFS遍历获取多跳邻居
    current_nodes = [node_idx]
    visited = set([node_idx])
    embedding_idx = 0
    
    for hop in range(use_hop + 1):
        next_level = []
        for node in current_nodes:
            neighbors = list(graph.neighbors(node))
            
            # 如果邻居太多，随机采样
            if len(neighbors) > sample_size:
                neighbors = random.sample(neighbors, sample_size)
            
            # 填充结构嵌入
            for i, neighbor in enumerate(neighbors):
                if embedding_idx < structure_dim:
                    structure_embedding[embedding_idx] = 1.0
                    embedding_idx += 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.append(neighbor)
        
        current_nodes = next_level
    
    return structure_embedding


def combine_embeddings(node_features, structure_embedding, embedding_type="simteg"):
    """
    组合不同类型的嵌入
    
    Args:
        node_features: 节点特征嵌入
        structure_embedding: 结构嵌入
        embedding_type: 嵌入类型 (sbert, roberta, e5, simteg)
        
    Returns:
        组合的嵌入
    """
    if embedding_type == "sbert":
        # 假设SBERT嵌入是384维
        return node_features[:384]
    elif embedding_type == "roberta":
        # 假设RoBERTa嵌入是1024维
        return node_features[:1024]
    elif embedding_type == "e5":
        # 假设E5嵌入是1024维
        return node_features[:1024]
    elif embedding_type == "simteg":
        # 组合SBERT(384)和RoBERTa/E5(1024*2)
        combined = np.concatenate([node_features, structure_embedding])
        return combined
    else:
        return node_features


def train_graph_node_model(
    embeddings, 
    texts, 
    embedding_dim, 
    lm_model_name="Vicuna-7b", 
    batch_size=16, 
    num_epochs=1,
    learning_rate=2e-3,
    projection_type="linear",
    projection_dim=None,
    freeze_backbone=True,
    use_wandb=True,
    wandb_project="graph_node_interpretation",
    bf16=True,
    deepspeed_config="zero2.json"
):
    """
    训练图节点解释模型
    
    Args:
        embeddings: 节点嵌入
        texts: 节点文本描述
        embedding_dim: 嵌入维度
        lm_model_name: 语言模型名称
        batch_size: 训练批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        projection_type: 投影类型 ("linear" 或 "2-layer-mlp")
        projection_dim: 投影层维度
        freeze_backbone: 是否冻结语言模型参数
        use_wandb: 是否使用W&B记录
        wandb_project: W&B项目名称
        bf16: 是否使用BF16精度
        deepspeed_config: DeepSpeed配置文件路径
        
    Returns:
        训练好的模型和分词器
    """
    # 初始化wandb
    if use_wandb:
        wandb_config = {
            "lm_model": lm_model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "embedding_dim": embedding_dim,
            "projection_type": projection_type,
            "projection_dim": projection_dim,
            "freeze_backbone": freeze_backbone,
            "dataset_size": len(embeddings),
            "bf16": bf16
        }
        wandb.init(project=wandb_project, config=wandb_config)
    
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集和数据加载器
    dataset = GraphEmbeddingDataset(embeddings, texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = GraphNodeModel(
        embedding_dim=embedding_dim,
        lm_model_name=lm_model_name,
        projection_type=projection_type,
        projection_dim=projection_dim,
        freeze_backbone=freeze_backbone,
        bf16=bf16
    )
    
    # 设置优化器
    # 仅优化投影器和提示词嵌入
    optimizer_params = [
        {'params': model.projector.parameters(), 'lr': learning_rate},
        {'params': model.prompt_embeddings, 'lr': learning_rate * 5.0}
    ]
    optimizer = optim.AdamW(optimizer_params, weight_decay=0.0)
    
    # 学习率调度器 - 余弦衰减与预热
    num_training_steps = len(dataloader) * num_epochs
    num_warmup_steps = int(0.03 * num_training_steps)  # 3%预热
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    # 初始化DeepSpeed
    ds_config = None
    if deepspeed_config:
        with open(deepspeed_config, 'r') as f:
            import json
            ds_config = json.load(f)
    
    # 将模型包装在DeepSpeed中
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        lr_scheduler=scheduler,
        dist_init_required=True
    )
    
    # 记录模型
    if use_wandb:
        wandb.watch(model_engine, log="all")
    
    # 训练循环
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"轮次 {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # 将批次移到设备
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            # 前向传递
            outputs = model_engine(
                graph_embeddings=batch["embedding"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            
            # 反向传递
            model_engine.backward(loss)
            model_engine.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # 记录每个批次的损失
            if use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "step": step + epoch * len(dataloader),
                    "learning_rate": scheduler.get_last_lr()[0]
                })
        
        avg_loss = total_loss / len(dataloader)
        print(f"轮次 {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")
        
        # 记录每个epoch的平均损失
        if use_wandb:
            wandb.log({
                "epoch_avg_loss": avg_loss,
                "epoch": epoch
            })
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 使用DeepSpeed保存
            model_engine.save_checkpoint("./best_model")
            
            if use_wandb:
                wandb.log({"best_loss": best_loss})
    
    # 关闭wandb
    if use_wandb:
        wandb.finish()
    
    return model_engine, tokenizer


def prepare_node_descriptions(G, node_features, labels, categories, node_indices=None, template_type="ND"):
    """
    为节点准备描述
    
    Args:
        G: NetworkX图
        node_features: 节点特征
        labels: 节点标签
        categories: 类别名称列表
        node_indices: 要处理的节点索引列表（如果为None，则处理所有节点）
        template_type: 模板类型 ("ND" 代表节点中心设计)
        
    Returns:
        节点描述字典
    """
    if node_indices is None:
        node_indices = range(len(node_features))
    
    node_texts = {}
    
    for i in node_indices:
        # 获取节点的类别
        category = categories[labels[i]]
        
        # 获取节点的特征词
        word_indices = np.where(node_features[i] > 0)[0]
        
        # 简化版词表
        simplified_vocab = [f"关键词{j}" for j in range(node_features.shape[1])]
        
        # 选择前5个(或更少)非零特征作为关键词
        top_words = [simplified_vocab[idx] for idx in word_indices[:min(5, len(word_indices))]]
        
        # 获取节点的邻居数量
        neighbors = list(G.neighbors(i))
        num_neighbors = len(neighbors)
        
        # 根据模板类型创建描述
        if template_type == "ND":
            if "arxiv" in categories[0].lower() or "cora" in categories[0].lower() or "pubmed" in categories[0].lower():
                # 学术图的描述
                description = f"这是一篇关于{category}领域的论文，主题是{'、'.join(top_words)}。"
                description += f"该论文引用了{num_neighbors}篇其他论文。"
            else:
                # 产品图的描述
                description = f"这是一个可以分类为{category}的亚马逊产品。"
                description += f"它可以被描述为{'、'.join(top_words)}。"
                description += f"该产品与{num_neighbors}个其他产品相关。"
        else:
            # 默认描述
            description = f"节点类别: {category}，包含关键词: {', '.join(top_words)}。"
            description += f"连接了{num_neighbors}个邻居节点。"
        
        node_texts[f"node{i}"] = description
    
    return node_texts


def main():
    parser = argparse.ArgumentParser(description="图节点解释模型训练")
    
    # 模型配置
    parser.add_argument("--lm_model", type=str, default="/fs-computility/mabasic/shared/models/Qwen2.5-7B-Instruct", help="语言模型名称")
    parser.add_argument("--embedding_type", type=str, default="simteg", choices=["sbert", "roberta", "e5", "simteg"], help="嵌入类型")
    parser.add_argument("--projection_type", type=str, default="linear", choices=["linear", "2-layer-mlp"], help="投影类型")
    parser.add_argument("--projection_dim", type=int, default=None, help="投影维度")
    
    # 图处理配置
    parser.add_argument("--use_hop", type=int, default=2, help="跳数")
    parser.add_argument("--sample_size", type=int, default=10, help="每个节点采样的邻居数")
    parser.add_argument("--template_type", type=str, default="ND", help="描述模板类型")
    
    # 训练配置
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--freeze_backbone", action="store_true", help="是否冻结主干网络")
    parser.add_argument("--bf16", action="store_true", help="使用BF16精度")
    
    # DeepSpeed配置
    parser.add_argument("--deepspeed_config", type=str, default="zero2.json", help="DeepSpeed配置文件")
    
    # W&B配置
    parser.add_argument("--use_wandb", action="store_true", help="是否使用W&B")
    parser.add_argument("--wandb_project", type=str, default="graph_node_interpretation", help="W&B项目名称")
    
    args = parser.parse_args()
    
    args.bf16 = True
    # 加载CORA数据集
    print("正在加载CORA数据集...")
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    print("加载数据集完成")
    
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
    
    # 准备节点描述
    node_texts = prepare_node_descriptions(
        G, node_features, labels, paper_categories, 
        template_type=args.template_type
    )
    
    print(f"已处理{len(node_embeddings)}个节点")
    
    # 确定嵌入维度
    embedding_dim = list(node_embeddings.values())[0].shape[0]
    print(f"最终嵌入维度: {embedding_dim}")
    
    # 打印一些样例
    sample_indices = random.sample(range(num_nodes), min(3, num_nodes))
    for idx in sample_indices:
        node_key = f"node{idx}"
        print(f"\n示例节点 {node_key}:")
        print(f"类别: {paper_categories[labels[idx]]}")
        print(f"描述: {node_texts[node_key]}")
    
    # 训练模型
    print("\n开始训练模型...")
    model, tokenizer = train_graph_node_model(
        node_embeddings,
        node_texts,
        embedding_dim,
        lm_model_name=args.lm_model,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        projection_type=args.projection_type,
        projection_dim=args.projection_dim,
        freeze_backbone=args.freeze_backbone,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        bf16=args.bf16,
        deepspeed_config=args.deepspeed_config
    )
    
    # 测试生成描述
    device = next(model.parameters()).device
    
    # 随机选择一个节点进行测试
    test_idx = random.choice(range(num_nodes))
    test_node_key = f"node{test_idx}"
    test_embedding = torch.tensor(node_embeddings[test_node_key], dtype=torch.bfloat16).to(device)
    
    print(f"\n测试节点: {test_node_key}")
    print(f"原始描述: {node_texts[test_node_key]}")
    
    # 进行推理
    model.eval()
    description = model.module.generate_description(test_embedding, tokenizer)
    print(f"生成的描述: {description}")
    
    print("\n模型训练和测试完成")

if __name__ == "__main__":
    main()