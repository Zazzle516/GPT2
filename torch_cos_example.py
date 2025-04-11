import torch

# Q1: 这个 forward 和 setup_context 的设计结构是什么  有什么样的调用顺序
# Q2: 这里 forward 返回的输入和输出在后续的梯度计算是怎么被利用的
# Q3: 在 setup_context 中传入的 ctx 是什么  有包含什么关键的信息吗  这个可以在 torch 源码中其他类似的例子找到
class Cosine(torch.autograd.Function):
    @staticmethod
    def forward(x0):
        x1 = torch.cos(x0)
        return x0, x1

    @staticmethod
    def setup_context(ctx, inputs, output):
        x1, x0 = output
        ctx.save_for_backward(x0)   # 额外保留输入

    @staticmethod
    def backward(ctx, grad_output):
        x0 = ctx.saved_tensors
        result = (-torch.sin(x0)) * grad_output     # 这里传进来的梯度是链式求导得到的吗
        return result

def f(x):
    x = torch.cos(x)
    x = torch.cos(x)
    return x

x = torch.randn(1024, 1024, 1024)
x.requires_grad_(True)
output = f(x)
output.sum().backward()