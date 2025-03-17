from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# 首先 从理论上 GPT-2 是 decoder-only model
# 相比于原始 Transformer 有两个改动
    # 1. LayerNorm 的计算位置
    # 2. 在最后的 SelfAttention Block 中额外添加了 layer normalization
# 在子模块命名中严格遵守了 HuggingFace 原本的 weight name

# Q: 目前 GPT-2 完整的结构已经实现  大约 100 行  但是 modeling_gpt2.py 的 2k+ 行代码优化了哪些地方

@dataclass
class GPTConfig:
    block_size: int = 1024         # 上下文长度
    vocab_size: int = 50527        # 模型可以识别的 token 总数
    n_embd:     int = 768          # 每个 token 的特征维度数

    n_layer:    int = 12           # Transformer.Block 中的 layer 数量
    n_head:     int = 12           # MHA 的切分数

# Multi-Layer Perceptron / Feed-Forward Network (FFN) = Linear + activateFunc + Linear
class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Tip: 这里 Linear 的输出神经元数量是 4×
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # activateFunc = GELU 针对 erf 计算近似简化
        # 官方在 huggingface.configuration_gpt2 中使用了 gelu_new line_145
        self.gelu = nn.GELU(approximate='tanh')

        # 对应到输入的 Linear 这里是 4× 的神经元输入数量
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Masked MHA
class CausalSelfAttention(nn.Module):
    # x.shape        = (BatchSize, SequenceLength, n_embd)
    # y.single_shape = (BatchSize, n_head        , SequenceLenght, head_dim = n_embd // n_head)
    # y.shape        = (batchSize, SequencceLenght, n_embd)

    def __init__(self, config: GPTConfig):
        super().__init__()
        # 检查当前的维度数能否平均分到多头任务上
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # 利用一个输入 x 根据 Wq, Wk, Wv 同时计算出 QKV  所以结果是一个 concated 的状态
        # 这里的 QKV 顺序是 openAI 提供的权重顺序决定的  和输入与计算方式无关
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)

        # 虽然名称是 bias 这是根据它的实际功能决定的  并不是来自于形状  提供掩码实现自注意
        # 注意最后调用的 view 形状调整  在 Transformer 中计算的 bias 是 [1, 1, T, T] 的形状
        # 所以 虽然数据内容是相同的 但要进行从 [T, T] => [1, 1, T, T] 的调整
        self.register_buffer("bias", torch.tril(
                                        torch.ones(
                                            config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        # 在把多个单头的计算输出整和到目标形状后  在内容层面赋予意义
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

    def forward(self, x):
        B, T, C = x.size()  # C = embedding dimensions = n_head * head_size

        # 此时 concat_qkv 继承 x 的形状  (B, T, C)
        concat_qkv = self.c_attn(x)
        q, k, v = concat_qkv.split(self.n_embd, dim=2)

        # 将一个完整的 QKV 矩阵均匀的划分到每个头执行  (B, nh, T, hs)
        # 通过 transpose(1, 2) 把每个头的数据独立出来  这样才可以进行多头的并行执行
        # 否则 如果是 (B, T, n_head, head_size) 的形状  T 在 n_head 前面  数据是混杂的
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Y = QK^T / sqrt(d_k)  d_k = head_size
        # 注意在计算 QK^T 的时候  K 张量需要转置  所以这里要交换 (-2, -1) 的位置
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 把 bias 中为 0 的部分变为 -inf  再经历 softmax 会变为 0
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        # 把单头的计算结果重新整合到 (B, T, C) 的形状  这里整合到 BTC 形状的前提是 T 和 n_head 的顺序
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