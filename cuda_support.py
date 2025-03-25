import torch
device = "cpu"      # 默认使用 CPU

# 如果有 cuda 用 cuda
if torch.cuda.is_available():
    device = "cuda"

# 判断是否是 Apple 系统 GPU=MPS(Metal Performance Shaders)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")


print(torch.cuda.current_device())  # 0

x = torch.tensor([1, 2, 3])
x = x.to("cuda")
print(x.device)

# https://github.com/karpathy/char-rnn