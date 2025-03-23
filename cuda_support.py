import torch
print(torch.cuda.is_available())

print(torch.cuda.current_device())  # 0

x = torch.tensor([1, 2, 3])
x = x.to("cuda")
print(x.device)