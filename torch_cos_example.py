import torch

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
        x0, _ = ctx.saved_tensors
        result = (-torch.sin(x0)) * grad_output     # 这里传进来的梯度是 LOSS 或者说上一步的 backward 结果
        return result

def f(x):
    x = torch.cos(x)
    x = torch.cos(x)
    return x

x = torch.randn(1024, 1024, 1024)
x.requires_grad_(True)
output = f(x)
output.sum().backward()


# 融合后算子代码
class OptimizedTwoCosine(torch.autograd.Function):
    @staticmethod
    def forward(x0):
        x1 = torch.cos(x0)
        x2 = torch.cos(x1)
        return x2, x0

    @staticmethod
    def setup_context(ctx, inputs, output):
        x2, x0 = output
        ctx.save_for_backward(x0)

    @staticmethod
    def backward(ctx, grad_x2):
        x0, _ = ctx.saved_tensors

        # 比起缓存 重新计算的效率更高
        x1 = torch.cos(x0)
        grad_x1 = (-torch.sin(x1)) * grad_x2
        grad_x0 = (-torch.sin(x0)) + grad_x1
        return grad_x0

def f2(x):
    x2, x0 = OptimizedTwoCosine.apply(x)
    return x2