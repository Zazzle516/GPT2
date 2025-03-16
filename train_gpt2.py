from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

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
    n_head:  int = 6            # MHA 的切分数

# Multi-Layer Perceptron / Feed-Forward Network (FFN) = Linear + activateFunc + Linear
class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Tip: 这里 Linear 的输出神经元数量是 4×
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # activateFunc = GELU 针对 erf 计算近似简化
        # 这里深入看了下  在 gelu_example.py 文件  至少在 huggingface.configuration_gpt2 中使用了 gelu_new
        # line_145  官方提供的也是 approx 版本的 gelu
        self.gelu = nn.GELU(approximate='tanh')

        # 对应到输入的 Linear 这里是 4× 的神经元输入数量
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        # 当前的维度数能否平均分到多头任务上
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # ???
        self.register_buffer("bias", torch.tril(
                                    torch.ones(
                                        config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # Batch_Size, 
        B, T, C = x.size()
        concat_qkv = self.c_attn(x)
        q, k, v = concat_qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Q: 得到自注意力矩阵 ???
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 通过 -inf 激活后变为 0
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y

class Block(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # ordered Layer
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)    # Masked MHA
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # Residual_Connection_1: A 数据流串行经过 layer_norm 和 Masked MHA
        # 再与无计算的数据流 B 计算  对比原本的 Transformer 结构  能更好的传递特征
        x = self.attn(self.ln_1(x))

        # Residual_Connection_2: 同理，A 数据流串行经过 layer_norm 和 FFN
        x = x + self.mlp(self.ln_2(x))
        return x


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