import torch
import time

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_texpr_fuser_enabled(False)

# 准确的 GELU 实现
@torch.jit.script
def gelu(x):
    return x * (0.5 + torch.erf(x * 0.7071067811865476) * 0.5)

# 使用 tanh 近似的 GELU 实现
@torch.jit.script
def fast_gelu_1(x):
    # sqrt(2/pi) = 0.7978845608028654
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3.0))))

# 精度更差但更快的 tanh 近似的 GELU 实现
@torch.jit.script
def fast_gelu_2(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

# 根据 https://github.com/pytorch/pytorch/issues/39853 链接给出的速度结果
# 大概是 gelu < fast_gelu_2 < fast_gelu_1 ???

# 首先尝试在本地用 CPU 测试一下

x = torch.randn(32, 128)
def time_me(fn):
    start_time = time.time()
    for i in range(100):
        fn(x)
    end_time = time.time()
    print(f"Execution time {end_time - start_time:.6f} seconds")

time_me(gelu)
time_me(fast_gelu_1)
time_me(fast_gelu_2)

# Execution time 0.013260 seconds
# Execution time 0.004061 seconds
# Execution time 0.003486 seconds

# 很难想象 但是结果确实是这个评论给出的顺序


# 再在 GPU 上测试一下



# Q: ReLU 的激活有问题吗
# 主要问题是在 [-1, 0] 这部分