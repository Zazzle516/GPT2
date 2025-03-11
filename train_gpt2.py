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
    block_size: int = 256       # 上下文长度
    vocab_size: int = 65        # 模型可以识别的 token 总数
    n_embd:  int = 384          # 每个 token 的特征维度数

    n_layer: int = 6            # Transformer.Block 中的 layer 数量
    n_head:  int = 6            # MHA 的并行程度 ?


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 完成 GPT-2 的网络骨架  根据标准 GPT2 的权重参数输出来仿照实现
        self.transformer = nn.ModuleDict(dict(
            # weight of the token embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),

            # weight of the position embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),

            # 这里是 GPT2 一共有 12 个 Transformer Block
            # 因为是通过 .h.0 => .h.11 来进行索引  所以使用 ModuleList 而不是 ModuleDict
            # 每个 Transformer Block 有 6 个 layer => ln_1, attn.c_atten, attn.c_proj, ln_2, mlp.c_fc, mlp.c_proj
            # 每层独立 weight 和 bias 所以总计是 12 个参数
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),

            # 针对 Transformer 而言的最后的 layer_norm_final
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # Language Module Head 用于输出 logits 预测下一个 token 的概率
        # 映射回 embedding 矩阵得到 token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)