from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# 在一些层中 share pareameters => 参数绑定  重点是该子模块的内部参数数值相同
# 直接通过赋值导出就可以了  很简单  而不是直接声明一个 比如说 nn.Linear() 这样
# 以引用的方式访问的  共享数据  通过子模块 A 改数据的话  B 模块的数据也会跟着改

# 不论是层还是 module 都是 nn.Module 的子类  没差啦  我感觉定义层还简单点...

# nn.Parameter 告知 pytorch 这些参数是可训练的  也就是计算反向传播
# 会自动被 model.parameters() 和 state_dict() 识别
# 如果把 weight_tensor 打印出来 对应的 requires_grad=true

# 首先 从理论上 GPT-2 是 decoder-only model
# 相比于原始 Transformer 有两个改动
    # 1. LayerNorm 的计算位置
    # 2. 在最后的 SelfAttention Block 中额外添加了 layer normalization

@dataclass
class GPTConfig:
    block_size: int = 256       # 上下文长度
    vocab_size: int = 65        # 模型可以识别的 token 总数
    n_embd:  int = 384          # 每个 token 的特征维度数

    n_layer: int = 6            # 
    n_head:  int = 6            # MHA 的并行程度 ?
    

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # init 并不是一个数据结构而是一个名称为 init.py 的文件
        # 并不是类似于 C 那样的文本导入 而是模块化管理
        # 注意 init.py 和 __init__.py 是两个文件
        # 通过 from . import init 这样导入文件  以作为参数的方式使用 ?
        # 怎么分辨自己可以被 nn 调用呢
        # 通过被 __init__.py 组织起来
        # nn.init.normal_()

        # 完成 GPT-2 的网络骨架  这些 API 有的涉及计算有的不涉及  但是统一通过 forward 来调用
        # 根据标准 GPT2 的权重参数输出来仿照实现

        # nn.Embedding(): 存放 Embedding 矩阵   param1: batch_size  param2: sequence_length
        self.transformer = nn.ModuleDict(dict(
            # weight of the token embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            # weight of the position embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),
            
            # 这里是 GPT2 一共有 12 个层 这边的 .h.0 => .h.11 来进行索引
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)