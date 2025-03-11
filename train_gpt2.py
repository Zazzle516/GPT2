from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F



@dataclass
class GPTConfig:
    # Q: 这些参数对应的什么 ?
    block_size: int = 256
    vocab_size: int = 65        # 上下文长度
    n_embd:  int = 384          # 每个 token 的特征维度数

    n_layer: int = 6
    n_head:  int = 6            # MHA 的并行程度
    

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # nn.ModuleDict() 是什么
        # nn.Embedding(): 存放 Embedding 矩阵   param1: batch_size  param2: sequence_length
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)