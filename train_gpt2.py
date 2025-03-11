from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# 首先 从理论上 GPT-2 是 decoder-only model
# 相比于原始 Transformer 有两个改动
    # 1. LayerNorm 的计算位置
    # 2. 在最后的 SelfAttention Block 中额外添加了 layer normalization

@dataclass
class GPTConfig:
    block_size: int = 256       # ?
    vocab_size: int = 65        # 上下文长度
    n_embd:  int = 384          # 每个 token 的特征维度数

    n_layer: int = 6
    n_head:  int = 6            # MHA 的并行程度
    

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 完成 GPT-2 的网络骨架  这些 API 有的涉及计算有的不涉及  但是统一通过 forward 来调用
        # nn.ModuleDict() 是什么
        # nn.Embedding(): 存放 Embedding 矩阵   param1: batch_size  param2: sequence_length
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)