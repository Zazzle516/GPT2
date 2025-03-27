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
# https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt2

# Q: 目前 GPT-2 完整的结构已经实现  大约 100 行  但是 modeling_gpt2.py 的 2k+ 行代码优化了哪些地方
# Q: 在这个 torch 实现的版本为什么会用到四个维度 ??

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
        # 通过 register_buffer 注册非训练参数  仍然会被 state_dict 发现并存储(persistent=True)  只是不参与训练而以
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

# 四个维度是在这里产生的  后续其他的参数也是因为 qkv 是四个维度才会 view()  所以为什么呢

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
        x = x + self.attn(self.ln_1(x))

        # Residual_Connection_2: 同理，A 数据流串行经过 layer_norm 和 FFN
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        # 如果直接传入 GPTConfig() 实例  那么 torch 会自动将这些参数随机化
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
        # [batch_size, sequence_length, n_embd]

        # Language Module Head 用于输出 logits 预测下一个 token 的概率
        # [batch_size, sequence_length, n_embd] @ [n_embd, vocab_size] 映射回 embedding 矩阵得到 token
        # [batch_size, sequence_length, vocab_size] 下一个 token 可能是全部 token 中的哪个
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

    # 使用 classmethod 返回 GPT 的实例 module line_179
    @classmethod
    def from_pretrained(cls, model_type):
        """Load pre-trained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # 通过索引 + 字典嵌套的方式高效查找   多层 key 值索引可以考虑这种方式
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt-large':    dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # 每个 model_type 对应一个 GPTConfig 的实例 把 config_args 的字典元素解包为 GPTConfig 的参数
        config = GPTConfig(**config_args)
        # 根据模型不同的参数  进行实例化
        model = GPT(config)

        # state_dict() 返回 orderedDict  Torch 自动管理的参数
        sd = model.state_dict()

        # 因为 HuggingFace 的 attn.bias 在 register_buffer.persistent=False 条件下并没有存储
        # 所以迁移的时候要筛选掉 Masked 部分
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # eg. line_114 => line_70

        # 得到 HF 的 GPT2 模型
        model_path = "/home/zazzle/models/" + model_type
        model_hf = GPT2LMHeadModel.from_pretrained(model_path)
        sd_hf = model_hf.state_dict()

        # 虽然 HuggingFace 没有存  但还是进行了删除操作  为了结构对齐
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # 把官方的权重参数数据拷到自己的模型参数里
        # 使用 no_grad() 防止 copy_() 被计算图追踪  减少梯度更新不必要的计算
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def forward(self, idx, targets=None):
        # idx: input index
        B, T = idx.size()   # T: Sequence Length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # 生成 Positional Embedding 的索引 [0, T-1]  stride 默认为 1
        # Tip: 这里只是索引 然后通过 wpe 和 wte 提供的 embedding 方法  查找到 Huggingace 提供的权重参数
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)     # [T, C]
        tok_emb = self.transformer.wte(idx)     # [B, T, C]

        # Q: 为什么这里通过相同 Embedding 结构获取到的参数形状不同呢
        # A: 注意两个索引的区别 pos=[0, 1, ...] 一维索引      idx=[B, T] 二维索引  pos 的所有 Batch 共享相同的位置编码
        x = tok_emb + pos_emb       # [T, C] => [B, T, C] + [B, T, C]

        # 顺序执行所有的 12 个 transformer block
        for block in self.transformer.h:
            x = block(x)    # 应该是和直接调用 block(x) 等价的 ??

        # 计算映射与归一化
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)    # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # targets [B, T]
            # logits.view(-1, logits.size(-1)) => [B * T, vocab_size]
            # targets.view(-1) => [B * T]
            # cross_entropy = -log( softmax(x)[y] ) = -log( exp(x[y]) / sum(exp(x)) )
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


device = 'cpu'

# create pre-token
import tiktoken
enc = tiktoken.get_encoding('gpt2')

# Inferance
# {
# num_return_sequence = 5
# max_length = 30

# tokens = enc.encode("Hello, I'm a language model,")

# 把 tokens 转换为 Tensor 来适配 torch  [sequence_length]
# Tip: 这里转换的只是索引 具体的 token_feature 在 line_225
# tokens = torch.tensor(tokens, dtype=torch.long)

# 1. unsqueeze(0): 在 idx=0 增加了一个维度 => 等效于 Batch=1 的输入 => 适配 Transformer 的计算形状 => line_50
# 2. repeat(N1, N2, ..): 在idx=0, idx=1, ... 的位置上重复 Ni 次 => 让 GPT2 模型最终生成 num_return_sequence 数量的结果
# tokens_idx = tokens_idx.unsqueeze(0).repeat(num_return_sequence, 1)

# [B=num_return_sequence(repeat)), T=sequence_lenght]  => line_218
# x = tokens_idx.to(device)

# generate right now  x=(B, T) B=5,T=8
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# model = GPT.from_pretrained('gpt2')
# model.eval()
# model.to(device)
# }

# Single Train
# {
# 把文本转换为 token 编码
# with open('DataSets/input.txt', 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens_idx = enc.encode(text)

# 定义 view 形状
B, T = 4, 32

# 把 tokens_idx 转换为 Tensor 来适配 torch  [sequence_length]
# Tip: 这里转换的只是索引 具体的 token_feature 在 line_225
# tokens_idx = torch.tensor(tokens_idx[:B * T + 1], dtype=torch.long)
# 必须手动获得在新内存上的指针
# tokens_idx = tokens_idx.to(device)

# 根据训练和预测不同的用途  进行偏移
# 此时学习前 1000 个词  只能得到过拟合的状态
# train_idx = tokens_idx[:-1].view(B, T)
# pred_idx = tokens_idx[1:].view(B, T)

# logits, loss = model(train_idx, pred_idx)
# print(loss)
# }

# Batch Train
# {
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # 加载完整的训练数据
        with open('DataSets/input.txt', 'r') as f:
            text = f.read()
        # enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loade {len(self.tokens)} tokens")                   # 打印 tokens 的总数
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")   # 单个 epoch 中的 token 数量

        # 初始化起始位置为 0
        self.current_position = 0
    
    def next_batch(self):
        # Tip: 注意这里需要预读一个 token
        B, T = self.B, self.T
        tokens_idx = self.tokens[self.current_position : self.current_position + B * T + 1]
        train_idx = (tokens_idx[:-1]).view(B, T)
        pred_idx = (tokens_idx[1:]).view(B, T)

        # 每次的 epoch 训练大小是 B*T  如果数据用完了那就循环回到开始
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return train_idx, pred_idx

train_loader = DataLoaderLite(B, T)
# }

model = GPT(GPTConfig())
model.to(device)


# 执行梯度下降
# Q: 这里的 Adam 和 AdamW 的区别是什么
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    train_idx, pred_idx = train_loader.next_batch()
    train_idx = train_idx.to(device)
    pred_idx = pred_idx.to(device)
    optimizer.zero_grad()
    logits, loss = model(train_idx, pred_idx)
    loss.backward()
    optimizer.step()
    print(f"train time {i}, loss: {loss.item()}")

import sys; sys.exit(0)

# 每个 loop 预测一个 token 然后拼接到已有序列中
# 每个 num_return_sequence 表示一个独立的生成序列
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)           # [batch_size, sequence_length, vocab_size]
        # num_return_sequence 会转换到 Batch 维度  每个 Batch 对应到一个生成序列
        # torch.multinomial 只抽取一个就好  因为抽取次数已经通过 Batch 定义了
        logits = logits[:, -1, :]   # [batch_size, vocab_size]  只需要最后一个 token 的概率分布来预测
        probs = F.softmax(logits, dim=-1)

        # topk_indices: 从 topk 中选出的前 50 个概率最高的 token ID
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # ix: 再从这 50 个 topk_indices 中选一个
        ix = torch.multinomial(topk_probs, 1)
        # xcol: 从 topk_indices 中取 ix 位置的 token ID
        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequence):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
